import os
import pathlib
import sys
import torch
import torch.nn as nn
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
from utils.model.simpleCNNbaseline import CNN_Transformer


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


def pearson_correlation_loss(y_pred, y_true):
    """
    计算皮尔逊相关系数损失
    Args:
        y_pred: 预测值 tensor
        y_true: 真实值 tensor
    Returns:
        loss: 1 - |r| 作为损失值（值越小相关性越强）
    """
    # 计算均值
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)

    # 计算协方差和方差
    cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true))
    var_pred = torch.mean((y_pred - mean_pred) ** 2)
    var_true = torch.mean((y_true - mean_true) ** 2)

    # 计算皮尔逊相关系数
    correlation = cov / (torch.sqrt(var_pred * var_true) + 1e-8)

    # 作为损失函数：1 - |r|，使相关性越强损失越小
    loss = 1 - torch.abs(correlation)

    return loss


class SequentialAccelerationSteeringLoss(nn.Module):
    def __init__(self, corr_weight=1.0, indep_weight=1.0, pearson_weight=1.0):
        super(SequentialAccelerationSteeringLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.corr_weight = corr_weight
        self.indep_weight = indep_weight
        self.pearson_weight = pearson_weight
        if corr_weight < 0 or indep_weight < 0 or pearson_weight < 0:
            raise ValueError(
                "Weights must be non-negative. Received: "
                f"corr_weight={corr_weight}, indep_weight={indep_weight}, pearson_weight={pearson_weight}"
            )
        
    def forward(self, pred_steering, pred_accel, target_steering, target_accel):
        """
        适配序列数据的损失函数
        pred_steering: [batch_size] - 预测的转向值
        pred_accel: [batch_size] - 预测的加速度值
        target_steering: [batch_size] - 目标转向值
        target_accel: [batch_size] - 目标加速度值
        """
        # 确保输入是1维张量
        pred_steering = pred_steering.view(-1)
        pred_accel = pred_accel.view(-1)
        target_steering = target_steering.view(-1)
        target_accel = target_accel.view(-1)

        # MSE loss for steering and acceleration
        steer_loss = self.mse(pred_steering, target_steering)
        accel_loss = self.mse(pred_accel, target_accel)

        # Correlation loss - 计算预测值之间的相关性
        corr_loss = correlation_loss(pred_steering, pred_accel)

        # Independence loss - 计算预测值之间的独立性
        indep_loss = independence_loss(pred_steering, pred_accel)

        # Pearson correlation loss - 计算预测值与目标值之间的相关性
        steer_pearson_loss = pearson_correlation_loss(
            pred_steering, target_steering
        )
        accel_pearson_loss = pearson_correlation_loss(pred_accel, target_accel)
        
        total_loss = (
            steer_loss
            + accel_loss
            + self.corr_weight * corr_loss
            + self.indep_weight * indep_loss
            + self.pearson_weight * (steer_pearson_loss + accel_pearson_loss)
        )

        return total_loss, steer_loss, accel_loss


class SequentialCNNDataset(Dataset):
    """
    Sequential CNN Dataset for time-series training

    Each batch contains sequential frames, and we predict the last frame's steering and acceleration
    """

    def __init__(self, root_dirs, batch_size=64, transform=None, split="train", frame_skip=1):
        """
        Args:
            root_dirs (list): List of dataset root directories (data_1, data_2, etc.)
            batch_size (int): Number of sequential frames in each batch
            transform (callable, optional): Image transformations
            split (str): 'train' or 'val'
            frame_skip (int): Skip every N frames (1 means skip every other frame for 10fps from 20fps)
        """
        self.transform = transform
        self.batch_size = batch_size
        self.frame_skip = frame_skip + 1  # +1 because we want to take every (frame_skip+1)th frame

        # Collect all image files from all data directories
        self.all_sequences = []

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

            # Apply frame skipping to reduce from 20fps to 10fps
            img_files = img_files[::self.frame_skip]
            
            # Create sequences of batch_size length
            img_paths = [os.path.join(data_dir, f) for f in img_files]

            # Create overlapping sequences
            for i in range(len(img_paths) - batch_size + 1):
                sequence = img_paths[i : i + batch_size]
                self.all_sequences.append(sequence)

        print(
            f"Created {split} dataset with {len(self.all_sequences)} sequences of length {batch_size} (frame_skip={frame_skip}, effective fps=10)"
        )

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        sequence_paths = self.all_sequences[idx]

        images = []
        speeds = []

        # Load all images and speeds in the sequence
        for img_path in sequence_paths:
            # Load image
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

            # Load speed from corresponding .txt file
            label_path = os.path.splitext(img_path)[0] + ".txt"
            with open(label_path, "r") as f:
                values = f.read().strip().split()
                speed = float(values[2])
            speeds.append(speed)

        # Load target (steering and acceleration) from the LAST frame
        last_label_path = os.path.splitext(sequence_paths[-1])[0] + ".txt"
        with open(last_label_path, "r") as f:
            values = f.read().strip().split()
            target_steering = float(values[0])
            target_acceleration = float(values[1])

        # Stack images and convert to tensors
        images = torch.stack(images)  # [batch_size, 3, H, W]
        speeds = torch.tensor(speeds, dtype=torch.float32)  # [batch_size]
        target_steering = torch.tensor(target_steering, dtype=torch.float32)  # 标量
        target_acceleration = torch.tensor(target_acceleration, dtype=torch.float32)  # 标量

        return images, speeds, target_steering, target_acceleration


def custom_collate_fn(batch):
    """
    Custom collate function to handle sequential data properly
    """
    # batch is a list of (images, speeds, steering, acceleration)
    # Each images is [batch_size, 3, H, W], speeds is [batch_size]

    all_images = []
    all_speeds = []
    all_steering = []
    all_acceleration = []

    for images, speeds, steering, acceleration in batch:
        all_images.append(images)
        all_speeds.append(speeds)
        all_steering.append(steering)
        all_acceleration.append(acceleration)

    # Stack everything
    all_images = torch.stack(all_images)  # [num_sequences, batch_size, 3, H, W]
    all_speeds = torch.stack(all_speeds)  # [num_sequences, batch_size]
    all_steering = torch.stack(all_steering)  # [num_sequences]
    all_acceleration = torch.stack(all_acceleration)  # [num_sequences]

    return all_images, all_speeds, all_steering, all_acceleration


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
    batch_size=64,
):
    """
    Train the Sequential CNN model with time-series data
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
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0
        train_accel_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_images, batch_speeds, steering_labels, accel_labels in progress_bar:
            batch_images = batch_images.to(
                device
            )  # [num_sequences, batch_size, 3, H, W]
            batch_speeds = batch_speeds.to(device)  # [num_sequences, batch_size]
            steering_labels = steering_labels.to(device)  # [num_sequences]
            accel_labels = accel_labels.to(device)  # [num_sequences]

            optimizer.zero_grad()

            sequence_steering_preds = []
            sequence_accel_preds = []

            # Process each sequence in the batch
            for seq_idx in range(batch_images.shape[0]):
                images_seq = batch_images[seq_idx]  # [batch_size, 3, H, W]
                speeds_seq = batch_speeds[seq_idx]  # [batch_size]

                # Forward pass through the sequence
                feature_queue = None
                for frame_idx in range(batch_size):
                    img = images_seq[frame_idx : frame_idx + 1]  # [1, 3, H, W]
                    speed = speeds_seq[frame_idx : frame_idx + 1]  # [1]

                    acc_output, steering_output, feature_queue = model(
                        img, speed, feature_queue
                    )

                # Only use the prediction from the last frame
                sequence_steering_preds.append(steering_output.squeeze())  # 标量
                sequence_accel_preds.append(acc_output.squeeze())  # 标量

            # Stack predictions
            pred_steering = torch.stack(sequence_steering_preds)  # [num_sequences]
            pred_accel = torch.stack(sequence_accel_preds)  # [num_sequences]

            # Calculate loss using custom sequential loss function
            total_loss, steer_loss, accel_loss = criterion(
                pred_steering, pred_accel, steering_labels, accel_labels
            )

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += total_loss.item()
            train_steering_loss += steer_loss.item()
            train_accel_loss += accel_loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "steer": f"{steer_loss.item():.4f}",
                    "accel": f"{accel_loss.item():.4f}",
                }
            )

        # Calculate average training loss
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
            for batch_images, batch_speeds, steering_labels, accel_labels in progress_bar:
                batch_images = batch_images.to(device)
                batch_speeds = batch_speeds.to(device)
                steering_labels = steering_labels.to(device)
                accel_labels = accel_labels.to(device)

                sequence_steering_preds = []
                sequence_accel_preds = []

                # Process each sequence in the batch
                for seq_idx in range(batch_images.shape[0]):
                    images_seq = batch_images[seq_idx]
                    speeds_seq = batch_speeds[seq_idx]

                    # Forward pass through the sequence
                    feature_queue = None
                    for frame_idx in range(batch_size):
                        img = images_seq[frame_idx : frame_idx + 1]
                        speed = speeds_seq[frame_idx : frame_idx + 1]

                        acc_output, steering_output, feature_queue = model(
                            img, speed, feature_queue
                        )

                    # Only use the prediction from the last frame
                    sequence_steering_preds.append(steering_output.squeeze())
                    sequence_accel_preds.append(acc_output.squeeze())

                # Stack predictions
                pred_steering = torch.stack(sequence_steering_preds)
                pred_accel = torch.stack(sequence_accel_preds)

                # Calculate loss using custom sequential loss function
                total_loss, steer_loss, accel_loss = criterion(
                    pred_steering, pred_accel, steering_labels, accel_labels
                )

                val_loss += total_loss.item()
                val_steering_loss += steer_loss.item()
                val_accel_loss += accel_loss.item()

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "val_loss": f"{total_loss.item():.4f}",
                        "val_steer": f"{steer_loss.item():.4f}",
                        "val_accel": f"{accel_loss.item():.4f}",
                    }
                )

        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_steering_loss /= len(val_loader)
        val_accel_loss /= len(val_loader)

        # Learning rate scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                str(model_save_path / "best_sequential_cnn_model.pth"),
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

        # Record history
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

        # Clear cache at the end of each epoch
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
    plt.savefig(str(model_save_path / "cnn_training_history.png"))
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

    model_save_path = ROOT_DIR / ".." / "run" / "sequential_cnn"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((640, 960)),  # 保持原始分辨率640*960
            transforms.ToTensor(),
        ]
    )

    # Model parameters
    batch_size = 40  # Sequence length
    dataloader_batch_size = 1  # Number of sequences per batch

    # 创建数据集和数据加载器，使用frame_skip=1来实现10fps采样
    train_dataset = SequentialCNNDataset(
        data_dirs, batch_size=batch_size, transform=transform, split="train", frame_skip=1
    )
    val_dataset = SequentialCNNDataset(
        data_dirs, batch_size=batch_size, transform=transform, split="val", frame_skip=1
    )

    # 数据加载器配置
    num_workers = min(16, os.cpu_count() or 2)  # Reduce workers for memory efficiency

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_batch_size,
        shuffle=True,  # Shuffle for better generalization
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # Print dataset info
    print(f"Training set size: {len(train_dataset)} sequences")
    print(f"Validation set size: {len(val_dataset)} sequences")
    print(f"Sequence length: {batch_size}")
    print(f"Dataloader batch size: {dataloader_batch_size}")

    # 创建Sequential CNN模型
    model = CNN_Transformer(
        input_len=2,
        seq_len=batch_size,
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        batch_size=1,  # Set to 1 for sequential processing
    )

    # 使用自定义序列损失函数，与train_CNN_transformer.py相同的权重配置
    criterion = SequentialAccelerationSteeringLoss(corr_weight=0.8, indep_weight=0.6, pearson_weight=2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.00005,
        weight_decay=1e-4,  # Slightly lower lr for sequential training
    )

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
        batch_size=batch_size,
    )

    # 绘制训练历史
    plot_training_history(history, model_save_path)

    # 保存最终模型
    torch.save(
        model.state_dict(), str(model_save_path / "final_sequential_cnn_model.pth")
    )
    print("Training completed, final model saved")
