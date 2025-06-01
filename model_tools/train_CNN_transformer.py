import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)  # Add utils directory to path
from utils.model.CNN_transformer import CNNViT


def correlation_loss(a, s):
    a = a.view(-1)
    s = s.view(-1)

    # 添加更严格的数值稳定性检查
    if len(a) < 2:
        return torch.tensor(0.0, device=a.device)

    a_mean = a.mean()
    s_mean = s.mean()

    # 检查标准差
    a_std = a.std()
    s_std = s.std()

    # 如果标准差太小，返回0损失
    if a_std < 1e-8 or s_std < 1e-8:
        return torch.tensor(0.0, device=a.device)

    cov = ((a - a_mean) * (s - s_mean)).mean()
    corr = cov / (a_std * s_std)

    # 限制相关系数范围并检查nan
    corr = torch.clamp(corr, -0.99, 0.99)
    if torch.isnan(corr) or torch.isinf(corr):
        return torch.tensor(0.0, device=a.device)

    return 1 - corr


def independence_loss(a, s):
    a = a.view(-1)
    s = s.view(-1)

    # 检查输入有效性
    if len(a) < 2 or torch.isnan(a).any() or torch.isnan(s).any():
        return torch.tensor(0.0, device=a.device)

    # 检查标准差
    if a.std() < 1e-8 or s.std() < 1e-8:
        return torch.tensor(0.0, device=a.device)

    try:
        corr_matrix = torch.corrcoef(torch.stack([a, s]))
        corr_val = corr_matrix[0, 1]

        if torch.isnan(corr_val) or torch.isinf(corr_val):
            return torch.tensor(0.0, device=a.device)

        return torch.abs(corr_val)
    except:
        # 如果计算失败，返回0
        return torch.tensor(0.0, device=a.device)


class AccelerationSteeringLoss(nn.Module):
    def __init__(self, corr_weight=1.0, indep_weight=1.0):
        super(AccelerationSteeringLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.corr_weight = corr_weight
        self.indep_weight = indep_weight

    def forward(self, pred, target):
        """
        pred: [batch_size, 2] - [steering, acceleration]
        target: [batch_size, 2] - [steering, acceleration]
        """
        steer_pred = pred[:, 0]
        accel_pred = pred[:, 1]
        steer_target = target[:, 0]
        accel_target = target[:, 1]

        # MSE loss for steering and acceleration
        steer_loss = self.mse(steer_pred, steer_target)
        accel_loss = self.mse(accel_pred, accel_target)

        # Correlation loss
        corr_loss = correlation_loss(steer_pred, accel_pred)

        # Independence loss
        indep_loss = independence_loss(steer_pred, accel_pred)

        total_loss = (
            steer_loss
            + accel_loss
            + self.corr_weight * corr_loss
            + self.indep_weight * indep_loss
        )

        return total_loss


class SimpleBaselineDataset(Dataset):
    """
    SimpleBaseline Dataset for frame-by-frame training (no sequences)

    Expected dataset structure:
    - Multiple datasets in data_n/train and data_n/val
    - Each image has a corresponding .txt file with: turing acceleration speed
    """

    def __init__(self, root_dirs, transform=None, split="train"):
        """
        Args:
            root_dirs (list): List of dataset root directories (data_1, data_2, etc.)
            transform (callable, optional): Image transformations
            split (str): 'train' or 'val'
        """
        self.transform = transform

        # Collect all image files from all data directories
        self.image_files = []
        for root_dir in root_dirs:
            data_dir = os.path.join(root_dir, split)
            if not os.path.exists(data_dir):
                print(f"Warning: Data directory {data_dir} not found, skipping.")
                continue

            # Get sorted image files to ensure sequential order
            img_files = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if f.endswith(".jpg") or f.endswith(".png")
                ],
                key=lambda x: int(os.path.splitext(x)[0]),
            )

            # Add path prefix to each file - use every frame (no subsampling)
            img_files = [os.path.join(data_dir, f) for f in img_files]
            self.image_files.extend(img_files)

        print(f"Created {split} dataset with {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Load image - 修改为RGB而不是灰度
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load label from corresponding .txt file
        label_path = os.path.splitext(img_path)[0] + ".txt"
        with open(label_path, "r") as f:
            values = f.read().strip().split()
            # Values format: turing acceleration speed
            turing = float(values[0])
            acceleration = float(values[1])
            speed = float(values[2])

        # Return single frame data - 修正数据格式
        steering = torch.tensor(turing, dtype=torch.float32)
        accel = torch.tensor(acceleration, dtype=torch.float32)
        speed_tensor = torch.tensor(speed, dtype=torch.float32)

        return image, steering, accel, speed_tensor


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device="cuda",
    model_save_path=None,
):
    """
    Train the SimpleBaseline model with frame-by-frame data
    """
    model = model.to(device)
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_steering_loss": [],
        "train_accel_loss": [],
        "val_steering_loss": [],
        "val_accel_loss": [],
    }

    # Early stopping parameters
    patience = 15
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0
        train_accel_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, steering_labels, accel_labels, speeds in progress_bar:
            images = images.to(device)
            steering_labels = steering_labels.to(device)
            accel_labels = accel_labels.to(device)
            speeds = speeds.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, speeds)

            # 组合目标标签
            targets = torch.stack([steering_labels, accel_labels], dim=1)

            # 计算损失
            total_loss = criterion(outputs, targets)

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 累积损失
            train_loss += total_loss.item()

            # 分别计算steering和acceleration的MSE损失用于监控
            with torch.no_grad():
                mse = nn.MSELoss()
                steer_loss = mse(outputs[:, 0], steering_labels)
                accel_loss = mse(outputs[:, 1], accel_labels)
                train_steering_loss += steer_loss.item()
                train_accel_loss += accel_loss.item()

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "steer": f"{steer_loss.item():.4f}",
                    "accel": f"{accel_loss.item():.4f}",
                }
            )

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_steering_loss /= len(train_loader)
        train_accel_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0
        val_accel_loss = 0.0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for images, steering_labels, accel_labels, speeds in progress_bar:
                images = images.to(device)
                steering_labels = steering_labels.to(device)
                accel_labels = accel_labels.to(device)
                speeds = speeds.to(device)

                # 前向传播
                outputs = model(images, speeds)

                # 组合目标标签
                targets = torch.stack([steering_labels, accel_labels], dim=1)

                # 计算损失
                total_loss = criterion(outputs, targets)
                val_loss += total_loss.item()

                # 分别计算steering和acceleration的MSE损失用于监控
                mse = nn.MSELoss()
                steer_loss = mse(outputs[:, 0], steering_labels)
                accel_loss = mse(outputs[:, 1], accel_labels)
                val_steering_loss += steer_loss.item()
                val_accel_loss += accel_loss.item()

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "val_loss": f"{total_loss.item():.4f}",
                        "val_steer": f"{steer_loss.item():.4f}",
                        "val_accel": f"{accel_loss.item():.4f}",
                    }
                )

        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_steering_loss /= len(val_loader)
        val_accel_loss /= len(val_loader)

        # 学习率调度器步进
        if scheduler:
            scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), str(model_save_path / "best_CNN_transformer_model.pth")
            )
            print(f"Model saved with validation loss: {val_loss:.4f}")
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print(
                    f"Early stopping: Validation loss didn't improve for {patience} epochs"
                )
                break

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_steering_loss"].append(train_steering_loss)
        history["train_accel_loss"].append(train_accel_loss)
        history["val_steering_loss"].append(val_steering_loss)
        history["val_accel_loss"].append(val_accel_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f} (Steer: {train_steering_loss:.4f}, Accel: {train_accel_loss:.4f}), "
            f"Val Loss: {val_loss:.4f} (Steer: {val_steering_loss:.4f}, Accel: {val_accel_loss:.4f})"
        )

        # 每个epoch结束清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return history


def plot_training_history(history, model_save_path=None):
    """Plot training history"""
    plt.figure(figsize=(15, 10))

    # Plot overall loss
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot steering loss
    plt.subplot(2, 2, 2)
    plt.plot(history["train_steering_loss"], label="Train Steering Loss")
    plt.plot(history["val_steering_loss"], label="Val Steering Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Steering Loss")
    plt.legend()
    plt.title("Steering Loss")

    # Plot acceleration loss
    plt.subplot(2, 2, 3)
    plt.plot(history["train_accel_loss"], label="Train Accel Loss")
    plt.plot(history["val_accel_loss"], label="Val Accel Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Acceleration Loss")
    plt.legend()
    plt.title("Acceleration Loss")

    plt.tight_layout()
    plt.savefig(str(model_save_path / "baseline_training_history.png"))
    plt.show()


if __name__ == "__main__":
    # 设置随机种子以提高可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # Define dataset paths - look for all data_n directories
    base_dir = ROOT_DIR / ".." / "dataset" / "CNNT"
    base_dir = base_dir.resolve()
    data_dirs = sorted(glob.glob(str(base_dir / "data_*")))

    if not data_dirs:
        print("No data directories found. Expected format: dataset/data_n/")
        sys.exit(1)

    print(f"Found {len(data_dirs)} data directories: {data_dirs}")

    model_save_path = ROOT_DIR / ".." / "run" / "CNN_transformer"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    transform = transforms.Compose(
        [
            transforms.Resize((640, 960)),  # 修正为正确的高度和宽度顺序
            transforms.ToTensor(),
        ]
    )

    # 创建数据集和数据加载器
    train_dataset = SimpleBaselineDataset(data_dirs, transform=transform, split="train")
    val_dataset = SimpleBaselineDataset(data_dirs, transform=transform, split="val")

    # 数据加载器配置
    num_workers = min(8, os.cpu_count() or 4)
    batch_size = 16  # 降低batch_size以适应transformer模型的内存需求

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Print dataset info
    print(f"Training set size: {len(train_dataset)} images")
    print(f"Validation set size: {len(val_dataset)} images")
    if len(train_dataset) > 0:
        print(f"Input image shape: {train_dataset[0][0].shape}")

    model = CNNViT()

    # 同步CNN的优化器配置
    criterion = AccelerationSteeringLoss(corr_weight=0.8,indep_weight=0.6)  # 使用自定义损失函数
    optimizer = optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=1e-4
    )  # 提高学习率到0.0001，权重衰减改为1e-4

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 训练模型
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        model_save_path=model_save_path,
    )

    # 绘制训练历史
    plot_training_history(history, model_save_path)

    # 保存最终模型
    torch.save(
        model.state_dict(), str(model_save_path / "final_CNN_transformer_model.pth")
    )
    print("Training completed, final model saved")
