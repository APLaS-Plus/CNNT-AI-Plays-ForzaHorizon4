import os
import pathlib
import sys
import torch
import torch.nn as nn
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
from utils.model.CNN_GLtransformer import CNNT


class CNNTDataset(Dataset):
    """
    CNNT Dataset for sequential training

    Expected dataset structure:
    - Multiple datasets in data_n/train and data_n/val
    - Each image has a corresponding .txt file with: turing acceleration speed
    """

    def __init__(self, root_dirs, transform=None, split="train", sequence_length=40):
        """
        Args:
            root_dirs (list): List of dataset root directories (data_1, data_2, etc.)
            transform (callable, optional): Image transformations
            split (str): 'train' or 'val'
            sequence_length (int): Number of frames to process as sequence
        """
        self.transform = transform
        self.sequence_length = sequence_length

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
            )  # Sort numerically by filename

            # Add path prefix to each file
            img_files = [os.path.join(data_dir, f) for f in img_files[::2]]
            self.image_files.extend(img_files)

        # Group images into sequences
        self.sequences = []
        for i in range(0, len(self.image_files) - sequence_length + 1, 1):
            self.sequences.append(self.image_files[i : i + sequence_length])

        print(f"Created {split} dataset with {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        images = []
        labels = []
        speeds = []

        for img_path in sequence:
            # Load image
            image = Image.open(img_path).convert("L")  # Grayscale
            if self.transform:
                image = self.transform(image)
            images.append(image)

            # Load label from corresponding .txt file
            label_path = os.path.splitext(img_path)[0] + ".txt"
            with open(label_path, "r") as f:
                values = f.read().strip().split()
                # Values format: turing acceleration speed
                turing = float(values[0])
                acceleration = float(values[1])
                speed = float(values[2])

                labels.append([turing, acceleration])
                speeds.append([speed])

        # Stack the sequence
        image_seq = torch.stack(images)  # shape: [seq_len, C, H, W]
        label_seq = torch.tensor(labels, dtype=torch.float32)  # shape: [seq_len, 2]
        speed_seq = torch.tensor(speeds, dtype=torch.float32)  # shape: [seq_len, 1]

        # 提取转向和加速度为单独的序列
        steering_seq = label_seq[:, 0:1]  # 取第一列并保持维度 [seq_len, 1]
        acceleration_seq = label_seq[:, 1:2]  # 取第二列并保持维度 [seq_len, 1]

        return image_seq, steering_seq, acceleration_seq, speed_seq


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=10,
    device="mps",  # 默认改为mps
    model_save_path=None,
    accumulation_steps=2,  # 梯度累积步数
    clip_grad_norm=1.0,  # 梯度裁剪阈值
    use_amp=False,  # Mac上默认不使用混合精度训练
):
    """
    Train the CNNT model with sequential data
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

    # MPS不支持混合精度训练，移除GradScaler
    scaler = None

    # Early stopping parameters
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0
        train_accel_loss = 0.0

        optimizer.zero_grad()
        total_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, steering_labels, accel_labels, speeds) in enumerate(
            progress_bar
        ):
            batch_size, seq_len = images.shape[0], images.shape[1]

            # 初始化队列
            frame_queue = None
            speed_queue = None
            steering_queue = None
            acceleration_queue = None
            memory_queue = None

            # 处理每个序列
            sequence_loss = 0.0
            sequence_steering_loss = 0.0
            sequence_accel_loss = 0.0

            # 移除混合精度上下文管理器
            for t in range(seq_len):
                current_frame = images[:, t].to(device)  # [B, C, H, W]
                current_speed = speeds[:, t].to(device)  # [B, 1]
                current_steering = steering_labels[:, t].to(device)  # [B, 1]
                current_accel = accel_labels[:, t].to(device)  # [B, 1]

                # 前向传播 - 使用新的模型接口
                (
                    steering,
                    acceleration,
                    new_memory,
                    frame_queue,
                    speed_queue,
                    steering_queue,
                    acceleration_queue,
                ) = model(
                    frame_queue,
                    speed_queue,
                    steering_queue,
                    acceleration_queue,
                    current_frame,
                    current_speed,
                    current_steering,
                    current_accel,
                    memory_tensor=memory_queue,
                    device=device,
                )

                # 更新记忆队列
                memory_queue = new_memory

                # 计算损失（跳过前10帧的预热阶段）
                if t >= 10:
                    target_steering = steering_labels[:, t, 0].to(device)
                    target_accel = accel_labels[:, t, 0].to(device)

                    steer_loss = criterion(steering, target_steering)
                    accel_loss = criterion(acceleration, target_accel)
                    frame_loss = steer_loss + accel_loss

                    sequence_loss += frame_loss.item()
                    sequence_steering_loss += steer_loss.item()
                    sequence_accel_loss += accel_loss.item()

            # 反向传播与优化
            if seq_len > 10:
                # 移除混合精度相关代码，直接执行标准的反向传播
                (frame_loss / accumulation_steps).backward()

                if ((batch_idx + 1) % accumulation_steps == 0) or (
                    batch_idx + 1 == total_batches
                ):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad_norm
                    )

                    # 更新参数
                    optimizer.step()
                    optimizer.zero_grad()

            # 计算平均损失
            avg_seq_loss = sequence_loss / max(1, seq_len - 10)
            avg_seq_steering_loss = sequence_steering_loss / max(1, seq_len - 10)
            avg_seq_accel_loss = sequence_accel_loss / max(1, seq_len - 10)

            train_loss += avg_seq_loss * batch_size
            train_steering_loss += avg_seq_steering_loss * batch_size
            train_accel_loss += avg_seq_accel_loss * batch_size

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "loss": f"{avg_seq_loss:.4f}",
                    "steer_loss": f"{avg_seq_steering_loss:.4f}",
                    "accel_loss": f"{avg_seq_accel_loss:.4f}",
                }
            )

            # 释放无用内存 - MPS不需要像CUDA那样明确清理缓存
            # if batch_idx % 5 == 0:
            #     torch.cuda.empty_cache()

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_steering_loss /= len(train_loader.dataset)
        train_accel_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0
        val_accel_loss = 0.0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for images, steering_labels, accel_labels, speeds in progress_bar:
                batch_size, seq_len = images.shape[0], images.shape[1]

                # 初始化队列
                frame_queue = None
                speed_queue = None
                steering_queue = None
                acceleration_queue = None
                memory_queue = None

                # 处理每个序列
                sequence_loss = 0.0
                sequence_steering_loss = 0.0
                sequence_accel_loss = 0.0

                for t in range(seq_len):
                    current_frame = images[:, t].to(device)
                    current_speed = speeds[:, t].to(device)
                    current_steering = steering_labels[:, t].to(device)
                    current_accel = accel_labels[:, t].to(device)

                    # 前向传播
                    (
                        steering,
                        acceleration,
                        new_memory,
                        frame_queue,
                        speed_queue,
                        steering_queue,
                        acceleration_queue,
                    ) = model(
                        frame_queue,
                        speed_queue,
                        steering_queue,
                        acceleration_queue,
                        current_frame,
                        current_speed,
                        current_steering,
                        current_accel,
                        memory_tensor=memory_queue,
                        device=device,
                    )

                    # 更新记忆队列
                    memory_queue = new_memory

                    # 计算损失（跳过前10帧）
                    if t >= 10:
                        target_steering = steering_labels[:, t, 0].to(device)
                        target_accel = accel_labels[:, t, 0].to(device)

                        steer_loss = criterion(steering, target_steering)
                        accel_loss = criterion(acceleration, target_accel)
                        frame_loss = steer_loss + accel_loss

                        sequence_loss += frame_loss.item()
                        sequence_steering_loss += steer_loss.item()
                        sequence_accel_loss += accel_loss.item()

                # 计算平均损失
                avg_seq_loss = sequence_loss / max(1, seq_len - 10)
                avg_seq_steering_loss = sequence_steering_loss / max(1, seq_len - 10)
                avg_seq_accel_loss = sequence_accel_loss / max(1, seq_len - 10)

                val_loss += avg_seq_loss * batch_size
                val_steering_loss += avg_seq_steering_loss * batch_size
                val_accel_loss += avg_seq_accel_loss * batch_size

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "val_loss": f"{avg_seq_loss:.4f}",
                        "val_steer": f"{avg_seq_steering_loss:.4f}",
                        "val_accel": f"{avg_seq_accel_loss:.4f}",
                    }
                )

        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_steering_loss /= len(val_loader.dataset)
        val_accel_loss /= len(val_loader.dataset)

        # 学习率调度器步进
        if scheduler:
            scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(model_save_path / "best_cnnt_model.pth"))
            print(f"Model saved with validation loss: {val_loss:.4f}")
            # 重置早停计数器
            counter = 0
        else:
            # 验证损失没有改善，增加计数器
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print(
                    "Early stopping: Validation loss didn't improve for {} epochs".format(
                        patience
                    )
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

        # 每个epoch结束清理缓存 - MPS不需要明确清理
        # torch.cuda.empty_cache()

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
    plt.savefig(str(model_save_path / "cnnt_training_history.png"))
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

    model_save_path = ROOT_DIR / ".." / "run" / "cnnt"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((240, 144)),
            transforms.ToTensor(),
        ]
    )

    # 创建数据集和数据加载器
    train_dataset = CNNTDataset(
        data_dirs, transform=transform, split="train", sequence_length=40
    )
    val_dataset = CNNTDataset(
        data_dirs, transform=transform, split="val", sequence_length=40
    )

    # 提高数据加载性能
    num_workers = min(8, os.cpu_count() or 4)  # 使用合适数量的工作线程
    prefetch_factor = 2  # 预取因子

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Print dataset info
    print(f"Training set size: {len(train_dataset)} sequences")
    print(f"Validation set size: {len(val_dataset)} sequences")
    if len(train_dataset) > 0:
        print(
            f"Input sequence shape: {train_dataset[0][0].shape}"
        )  # Should be [seq_len, 1, 240, 136]

    # 创建CNNT模型
    model = CNNT(input_height=240, input_width=144, maxtime_step=40, memory_size=150)

    # 使用更高级的优化器配置
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    # 检查MPS可用性
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    # Mac不支持混合精度训练
    use_amp = False
    
    # 训练模型
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        model_save_path=model_save_path,
        accumulation_steps=2,  # 使用梯度累积
        clip_grad_norm=1.0,  # 使用梯度裁剪
        use_amp=use_amp,  # Mac上不使用混合精度
    )

    # 绘制训练历史
    plot_training_history(history, model_save_path)

    # 保存最终模型
    torch.save(model.state_dict(), str(model_save_path / "final_cnnt_model.pth"))
    print("Training completed, final model saved")
