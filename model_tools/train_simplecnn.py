import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml
from datetime import datetime
import platform

# 替换tqdm为rich
from rich.progress import (
    Progress,
    TaskID,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

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
    var_true = torch.mean((y_true - mean_pred) ** 2)

    # 检查方差是否为零
    if var_pred < 1e-8 or var_true < 1e-8:
        return torch.tensor(0.0, device=y_pred.device)

    # 计算皮尔逊相关系数
    correlation = cov / (torch.sqrt(var_pred * var_true) + 1e-8)

    # 限制相关系数范围并检查nan
    correlation = torch.clamp(correlation, -0.99, 0.99)
    if torch.isnan(correlation) or torch.isinf(correlation):
        return torch.tensor(0.0, device=y_pred.device)

    # 作为损失函数：1 - r，使相关性越强损失越小
    loss = 1 - correlation

    return loss


class DerivativeLoss(nn.Module):
    """
    L = MSE(y, ŷ) + λ * MSE(Δy, Δŷ)
    其中 Δ 在时间维度上做差分：Δy_t = y_t - y_{t-1}
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred, y_true: [B, T]
        """
        # 差分 MSE（趋势）
        dy_pred = torch.diff(y_pred, dim=1)
        dy_true = torch.diff(y_true, dim=1)
        loss_trend = self.mse(dy_pred, dy_true)

        return loss_trend


class SequentialAccelerationSteeringLoss(nn.Module):
    def __init__(self, Derivative_derta=0.2, pearson_weight=0.5):
        super(SequentialAccelerationSteeringLoss, self).__init__()
        if Derivative_derta < 0 or pearson_weight < 0:
            raise ValueError(
                "Weights must be non-negative. Received: "
                f"Derivative_derta={Derivative_derta}, pearson_weight={pearson_weight}"
            )

        self.mse = nn.MSELoss()
        self.Derivative_derta = Derivative_derta
        self.derivativeLoss = DerivativeLoss()
        self.pearson_weight = pearson_weight

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

        # Derivative loss for acceleration
        steer_derivative_loss = self.Derivative_derta * self.derivativeLoss(
            pred_steering, target_steering
        )
        accel_derivative_loss = self.Derivative_derta * self.derivativeLoss(
            pred_accel, target_accel
        )
        derivative_loss = steer_derivative_loss + accel_derivative_loss

        # Pearson correlation loss - 计算预测值与目标值之间的相关性
        steer_pearson_loss = self.pearson_weight * pearson_correlation_loss(
            pred_steering, target_steering
        )
        accel_pearson_loss = self.pearson_weight * pearson_correlation_loss(
            pred_accel, target_accel
        )
        pearson_loss = steer_pearson_loss + accel_pearson_loss
        total_loss = steer_loss + accel_loss + derivative_loss + pearson_loss

        return (
            total_loss,
            steer_loss,
            accel_loss,
            steer_derivative_loss,
            accel_derivative_loss,
            steer_pearson_loss,
            accel_pearson_loss,
        )


class SequentialCNNDataset(Dataset):
    """
    Sequential CNN Dataset for time-series training

    Each batch contains sequential frames, and we predict the last frame's steering and acceleration
    """

    def __init__(
        self, root_dirs, batch_size=64, transform=None, split="train", frame_skip=1
    ):
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
        self.frame_skip = (
            frame_skip + 1
        )  # +1 because we want to take every (frame_skip+1)th frame

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
            img_files = img_files[:: self.frame_skip]

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
        target_acceleration = torch.tensor(
            target_acceleration, dtype=torch.float32
        )  # 标量

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
    console = Console()
    model = model.to(device)
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_steering_loss": [],
        "train_accel_loss": [],
        "val_steering_loss": [],
        "val_accel_loss": [],
        "train_derivative_loss": [],
        "train_pearson_loss": [],
        "val_derivative_loss": [],
        "val_pearson_loss": [],
    }

    # Early stopping parameters
    patience = 5
    counter = 0

    # 创建rich进度显示
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        expand=True,
    ) as progress:

        epoch_task = progress.add_task("[green]Training Progress", total=num_epochs)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_steering_loss = 0.0
            train_accel_loss = 0.0
            train_steer_derivative_loss = 0.0
            train_accel_derivative_loss = 0.0
            train_steer_pearson_loss = 0.0
            train_accel_pearson_loss = 0.0

            # 创建训练进度条
            train_task = progress.add_task(
                f"[cyan]Epoch {epoch+1}/{num_epochs} [Train]", total=len(train_loader)
            )

            for batch_idx, (
                batch_images,
                batch_speeds,
                steering_labels,
                accel_labels,
            ) in enumerate(train_loader):
                batch_images = batch_images.to(device)
                batch_speeds = batch_speeds.to(device)
                steering_labels = steering_labels.to(device)
                accel_labels = accel_labels.to(device)

                optimizer.zero_grad()

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
                (
                    total_loss,
                    steer_loss,
                    accel_loss,
                    steer_derivative_loss,
                    accel_derivative_loss,
                    steer_pearson_loss,
                    accel_pearson_loss,
                ) = criterion(
                    pred_steering,
                    pred_accel,
                    steering_labels,
                    accel_labels,
                )

                # Backward pass
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Accumulate losses
                train_loss += total_loss.item()
                train_steering_loss += steer_loss.item()
                train_accel_loss += accel_loss.item()
                train_steer_derivative_loss += steer_derivative_loss.item()
                train_accel_derivative_loss += accel_derivative_loss.item()
                train_steer_pearson_loss += steer_pearson_loss.item()
                train_accel_pearson_loss += accel_pearson_loss.item()

                # 更新进度条并显示详细信息
                progress.update(
                    train_task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch+1} [Train] - "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Steer: {steer_loss.item():.4f} | "
                    f"Accel: {accel_loss.item():.4f} | "
                    f"Deriv: {(steer_derivative_loss.item() + accel_derivative_loss.item()):.4f} | "
                    f"Pearson: {(steer_pearson_loss.item() + accel_pearson_loss.item()):.4f}",
                )

            # 移除训练进度条
            progress.remove_task(train_task)

            # Calculate average training loss
            train_loss /= len(train_loader)
            train_steering_loss /= len(train_loader)
            train_accel_loss /= len(train_loader)
            train_steer_derivative_loss /= len(train_loader)
            train_accel_derivative_loss /= len(train_loader)
            train_steer_pearson_loss /= len(train_loader)
            train_accel_pearson_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_steering_loss = 0.0
            val_accel_loss = 0.0
            val_steer_derivative_loss = 0.0
            val_accel_derivative_loss = 0.0
            val_steer_pearson_loss = 0.0
            val_accel_pearson_loss = 0.0

            # 创建验证进度条
            val_task = progress.add_task(
                f"[magenta]Epoch {epoch+1}/{num_epochs} [Val]", total=len(val_loader)
            )

            with torch.no_grad():
                for batch_idx, (
                    batch_images,
                    batch_speeds,
                    steering_labels,
                    accel_labels,
                ) in enumerate(val_loader):
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
                    (
                        total_loss,
                        steer_loss,
                        accel_loss,
                        steer_derivative_loss,
                        accel_derivative_loss,
                        steer_pearson_loss,
                        accel_pearson_loss,
                    ) = criterion(
                        pred_steering,
                        pred_accel,
                        steering_labels,
                        accel_labels,
                    )

                    val_loss += total_loss.item()
                    val_steering_loss += steer_loss.item()
                    val_accel_loss += accel_loss.item()
                    val_steer_derivative_loss += steer_derivative_loss.item()
                    val_accel_derivative_loss += accel_derivative_loss.item()
                    val_steer_pearson_loss += steer_pearson_loss.item()
                    val_accel_pearson_loss += accel_pearson_loss.item()

                    # 更新验证进度条
                    progress.update(
                        val_task,
                        advance=1,
                        description=f"[magenta]Epoch {epoch+1} [Val] - "
                        f"Loss: {total_loss.item():.4f} | "
                        f"Steer: {steer_loss.item():.4f} | "
                        f"Accel: {accel_loss.item():.4f} | "
                        f"Deriv: {(steer_derivative_loss.item() + accel_derivative_loss.item()):.4f} | "
                        f"Pearson: {(steer_pearson_loss.item() + accel_pearson_loss.item()):.4f}",
                    )

            # 移除验证进度条
            progress.remove_task(val_task)

            # Calculate average validation loss
            val_loss /= len(val_loader)
            val_steering_loss /= len(val_loader)
            val_accel_loss /= len(val_loader)
            val_steer_derivative_loss /= len(val_loader)
            val_accel_derivative_loss /= len(val_loader)
            val_steer_pearson_loss /= len(val_loader)
            val_accel_pearson_loss /= len(val_loader)

            # Learning rate scheduler step
            if scheduler:
                scheduler.step(val_loss)

            # 创建详细的损失表格
            table = Table(title=f"Epoch {epoch+1} Results")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Train", style="green")
            table.add_column("Val", style="magenta")
            table.add_column("Improvement", style="yellow")

            # 计算改进
            prev_val_loss = (
                history["val_loss"][-1] if history["val_loss"] else float("inf")
            )
            improvement = "↓" if val_loss < prev_val_loss else "↑"

            table.add_row(
                "Total Loss", f"{train_loss:.4f}", f"{val_loss:.4f}", improvement
            )
            table.add_row(
                "Steering Loss",
                f"{train_steering_loss:.4f}",
                f"{val_steering_loss:.4f}",
                "",
            )
            table.add_row(
                "Accel Loss", f"{train_accel_loss:.4f}", f"{val_accel_loss:.4f}", ""
            )
            table.add_row(
                "Steer Derivative",
                f"{train_steer_derivative_loss:.4f}",
                f"{val_steer_derivative_loss:.4f}",
                "",
            )
            table.add_row(
                "Accel Derivative",
                f"{train_accel_derivative_loss:.4f}",
                f"{val_accel_derivative_loss:.4f}",
                "",
            )
            table.add_row(
                "Steer Pearson",
                f"{train_steer_pearson_loss:.4f}",
                f"{val_steer_pearson_loss:.4f}",
                "",
            )
            table.add_row(
                "Accel Pearson",
                f"{train_accel_pearson_loss:.4f}",
                f"{val_accel_pearson_loss:.4f}",
                "",
            )

            console.print(table)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    str(model_save_path / "best_sequential_cnn_model.pth"),
                )
                console.print(
                    f"[bold green]✅ Model saved with validation loss: {val_loss:.4f}[/bold green]"
                )
                counter = 0
            else:
                counter += 1
                console.print(
                    f"[yellow]⚠️  EarlyStopping counter: {counter} out of {patience}[/yellow]"
                )
                if counter >= patience:
                    console.print(
                        f"[bold red]🛑 Early stopping: Validation loss didn't improve for {patience} epochs[/bold red]"
                    )
                    break

            # Record history with more detailed metrics
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_steering_loss"].append(train_steering_loss)
            history["train_accel_loss"].append(train_accel_loss)
            history["val_steering_loss"].append(val_steering_loss)
            history["val_accel_loss"].append(val_accel_loss)
            history["train_derivative_loss"].append(
                train_steer_derivative_loss + train_accel_derivative_loss
            )
            history["train_pearson_loss"].append(
                train_steer_pearson_loss + train_accel_pearson_loss
            )
            history["val_derivative_loss"].append(
                val_steer_derivative_loss + val_accel_derivative_loss
            )
            history["val_pearson_loss"].append(
                val_steer_pearson_loss + val_accel_pearson_loss
            )

            # 更新总体epoch进度
            progress.update(epoch_task, advance=1)

            # Clear cache at the end of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return history


def plot_training_history(history, model_save_path=None):
    """Plot training history with more detailed metrics"""
    plt.figure(figsize=(20, 15))

    # Plot overall loss
    plt.subplot(3, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    # Plot steering loss
    plt.subplot(3, 3, 2)
    plt.plot(history["train_steering_loss"], label="Train Steering Loss", color="green")
    plt.plot(history["val_steering_loss"], label="Val Steering Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Steering Loss")
    plt.legend()
    plt.title("Steering Loss")
    plt.grid(True, alpha=0.3)

    # Plot acceleration loss
    plt.subplot(3, 3, 3)
    plt.plot(history["train_accel_loss"], label="Train Accel Loss", color="purple")
    plt.plot(history["val_accel_loss"], label="Val Accel Loss", color="brown")
    plt.xlabel("Epoch")
    plt.ylabel("Acceleration Loss")
    plt.legend()
    plt.title("Acceleration Loss")
    plt.grid(True, alpha=0.3)

    # Plot derivative loss
    plt.subplot(3, 3, 4)
    plt.plot(
        history["train_derivative_loss"], label="Train Derivative Loss", color="cyan"
    )
    plt.plot(
        history["val_derivative_loss"], label="Val Derivative Loss", color="magenta"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Derivative Loss")
    plt.legend()
    plt.title("Derivative Loss")
    plt.grid(True, alpha=0.3)

    # Plot Pearson loss
    plt.subplot(3, 3, 5)
    plt.plot(history["train_pearson_loss"], label="Train Pearson Loss", color="olive")
    plt.plot(history["val_pearson_loss"], label="Val Pearson Loss", color="navy")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson Loss")
    plt.legend()
    plt.title("Pearson Correlation Loss")
    plt.grid(True, alpha=0.3)

    # Loss comparison bar chart for final epoch
    plt.subplot(3, 3, 6)
    final_losses = [
        history["train_loss"][-1],
        history["val_loss"][-1],
        history["train_steering_loss"][-1],
        history["val_steering_loss"][-1],
        history["train_accel_loss"][-1],
        history["val_accel_loss"][-1],
    ]
    loss_labels = [
        "Train Total",
        "Val Total",
        "Train Steer",
        "Val Steer",
        "Train Accel",
        "Val Accel",
    ]
    plt.bar(
        loss_labels,
        final_losses,
        color=["blue", "red", "green", "orange", "purple", "brown"],
    )
    plt.ylabel("Loss Value")
    plt.title("Final Epoch Loss Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Training vs Validation loss ratio
    plt.subplot(3, 3, 7)
    train_val_ratio = [
        t / v if v > 0 else 1
        for t, v in zip(history["train_loss"], history["val_loss"])
    ]
    plt.plot(train_val_ratio, label="Train/Val Ratio", color="darkred")
    plt.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="Perfect Ratio")
    plt.xlabel("Epoch")
    plt.ylabel("Train/Val Loss Ratio")
    plt.legend()
    plt.title("Train/Validation Loss Ratio")
    plt.grid(True, alpha=0.3)

    # Loss components stacked area chart
    plt.subplot(3, 3, 8)
    epochs = range(len(history["train_loss"]))
    plt.stackplot(
        epochs,
        history["train_steering_loss"],
        history["train_accel_loss"],
        history["train_derivative_loss"],
        history["train_pearson_loss"],
        labels=["Steering", "Acceleration", "Derivative", "Pearson"],
        alpha=0.7,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss Components")
    plt.legend(loc="upper right")
    plt.title("Training Loss Components (Stacked)")
    plt.grid(True, alpha=0.3)

    # Learning curve smoothed
    plt.subplot(3, 3, 9)
    # Simple moving average for smoothing
    window = 5
    if len(history["train_loss"]) >= window:
        train_smooth = np.convolve(
            history["train_loss"], np.ones(window) / window, mode="valid"
        )
        val_smooth = np.convolve(
            history["val_loss"], np.ones(window) / window, mode="valid"
        )
        epochs_smooth = range(window - 1, len(history["train_loss"]))
        plt.plot(
            epochs_smooth,
            train_smooth,
            label="Train Loss (Smoothed)",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            epochs_smooth,
            val_smooth,
            label="Val Loss (Smoothed)",
            color="red",
            linewidth=2,
        )
    else:
        plt.plot(history["train_loss"], label="Train Loss", color="blue")
        plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Smoothed Loss")
    plt.legend()
    plt.title("Smoothed Learning Curves")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        str(model_save_path / "detailed_cnn_training_history.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def save_training_config(
    model_save_path,
    model_config,
    training_config,
    optimizer_config,
    scheduler_config,
    dataset_config,
    loss_config,
    hardware_config,
    history,
):
    """
    保存训练配置到YAML文件
    """
    config = {
        "training_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
        },
        "model_config": model_config,
        "training_config": training_config,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
        "dataset_config": dataset_config,
        "loss_config": loss_config,
        "hardware_config": hardware_config,
        "training_results": {
            "total_epochs": len(history["train_loss"]),
            "best_val_loss": float(min(history["val_loss"])),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_train_steering_loss": float(history["train_steering_loss"][-1]),
            "final_val_steering_loss": float(history["val_steering_loss"][-1]),
            "final_train_accel_loss": float(history["train_accel_loss"][-1]),
            "final_val_accel_loss": float(history["val_accel_loss"][-1]),
            "final_train_derivative_loss": (
                float(history["train_derivative_loss"][-1])
                if history["train_derivative_loss"]
                else 0.0
            ),
            "final_val_derivative_loss": (
                float(history["val_derivative_loss"][-1])
                if history["val_derivative_loss"]
                else 0.0
            ),
            "final_train_pearson_loss": (
                float(history["train_pearson_loss"][-1])
                if history["train_pearson_loss"]
                else 0.0
            ),
            "final_val_pearson_loss": (
                float(history["val_pearson_loss"][-1])
                if history["val_pearson_loss"]
                else 0.0
            ),
        },
    }

    # 保存配置文件
    config_path = (
        model_save_path
        / f"training_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)

    console = Console()
    console.print(
        f"[bold green]📄 Training configuration saved to: {config_path}[/bold green]"
    )

    return config_path


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
    frame_skip = 3  # Skip every other frame to achieve 5 fps from 20 fps
    batch_size = 40  # Sequence length
    dataloader_batch_size = 1  # Number of sequences per batch
    num_epochs = 100  # Number of training epochs
    steps_per_epoch = len(data_dirs) * (
        len(os.listdir(data_dirs[0])) // (frame_skip + 1) // batch_size
    )
    warmup_steps = 500
    total_steps = num_epochs * steps_per_epoch
    
    print(f"Using frame skip: {frame_skip} to achieve 10fps from 20fps")
    print(f"Batch size: {batch_size}, Dataloader batch size: {dataloader_batch_size}")
    print(f"Number of epochs: {num_epochs}, Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # 创建数据集和数据加载器
    train_dataset = SequentialCNNDataset(
        data_dirs,
        batch_size=batch_size,
        transform=transform,
        split="train",
        frame_skip=frame_skip,
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

    def setup_layered_optimizer(
        model, cnn_lr=1e-4, transformer_lr=5e-5, weight_decay=1e-4
    ):
        """
        为模型的不同部分设置不同的学习率
        Args:
            model: CNN_Transformer模型
            cnn_lr: CNN部分的学习率
            transformer_lr: Transformer部分的学习率
            weight_decay: 权重衰减
        Returns:
            optimizer: 配置好的优化器
        """
        # 定义参数组
        cnn_params = []
        transformer_params = []

        # 遍历模型参数，按模块分类
        for name, param in model.named_parameters():
            if param.requires_grad:
                # CNN相关部分 - 特征提取器和CNN-ViT模块
                if any(
                    keyword in name
                    for keyword in [
                        "cnnvit.feature_extractor",
                        "cnnvit.image_encoder",
                        "cnnvit.speed_encoder",
                        "cnnvit.downsample",
                    ]
                ):
                    cnn_params.append(param)
                    print(f"CNN部分参数: {name}")

                # Transformer相关部分 - 时间序列Transformer和输出层
                elif any(
                    keyword in name
                    for keyword in [
                        "transformer_blocks",
                        "timepositional_embedding",
                        "timetransformer",
                        "downsample",
                        "acc_branch",
                        "steering_branch",
                        "combined_branch",
                        "acc_outputlayer",
                        "steering_outputlayer",
                    ]
                ):
                    transformer_params.append(param)
                    print(f"Transformer部分参数: {name}")

                else:
                    # 默认归类到CNN部分
                    transformer_params.append(param)
                    print(f"默认CNN部分参数: {name}")

        # 创建参数组
        param_groups = [
            {
                "params": cnn_params,
                "lr": cnn_lr,
                "weight_decay": weight_decay,
                "name": "cnn_features",
            },
            {
                "params": transformer_params,
                "lr": transformer_lr,
                "weight_decay": weight_decay,
                "name": "transformer_temporal",
            },
        ]

        return param_groups

    # 使用分层优化器
    cnn_lr = 1e-4
    transformer_lr = 5e-5
    weight_decay = 1e-4
    param_groups = setup_layered_optimizer(
        model, cnn_lr=cnn_lr, transformer_lr=transformer_lr, weight_decay=weight_decay
    )
    print("Parameter groups for optimizer:")
    for group in param_groups:
        print(
            f"Group: {group['name']}, Learning Rate: {group['lr']}, Params: {len(group['params'])}"
        )

    # 使用自定义序列损失函数配置
    derivative_delta = 0.7
    pearson_weight = 0.5
    criterion = SequentialAccelerationSteeringLoss(
        Derivative_derta=derivative_delta, pearson_weight=pearson_weight
    )
    optimizer = optim.AdamW(param_groups)

    # 先建两个调度器
    warmup = LinearLR(
        optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )

    # 添加学习率调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 收集配置信息用于保存
    model_config = {
        "model_type": "CNN_Transformer",
        "input_len": 2,
        "seq_len": batch_size,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "model_batch_size": 1,
    }

    training_config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "dataloader_batch_size": dataloader_batch_size,
        "frame_skip": frame_skip,
        "early_stopping_patience": 5,
        "gradient_clip_max_norm": 1.0,
        "random_seed": 42,
    }

    optimizer_config = {
        "optimizer_type": "AdamW",
        "cnn_learning_rate": cnn_lr,
        "transformer_learning_rate": transformer_lr,
        "weight_decay": weight_decay,
        "layered_optimization": True,
    }

    scheduler_config = {
        "scheduler_type": "SequentialLR",
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "warmup_start_factor": 0.2,
        "warmup_end_factor": 1.0,
        "cosine_eta_min": 1e-6,
    }

    dataset_config = {
        "dataset_type": "SequentialCNNDataset",
        "data_directories": [str(d) for d in data_dirs],
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset),
        "image_size": [640, 960],
        "num_workers": num_workers,
        "pin_memory": True,
        "shuffle_train": True,
    }

    loss_config = {
        "loss_function": "SequentialAccelerationSteeringLoss",
        "derivative_delta": derivative_delta,
        "pearson_weight": pearson_weight,
        "components": [
            "MSE_steering",
            "MSE_acceleration",
            "derivative_loss",
            "pearson_correlation_loss",
        ],
    }

    hardware_config = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
    }

    # 训练模型
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
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

    # 保存训练配置到YAML文件
    config_path = save_training_config(
        model_save_path=model_save_path,
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        hardware_config=hardware_config,
        history=history,
    )

    console = Console()
    console.print(f"[bold blue]🎉 Training completed successfully![/bold blue]")
    console.print(f"[green]📁 Model files saved to: {model_save_path}[/green]")
    console.print(f"[green]📋 Configuration saved to: {config_path}[/green]")
