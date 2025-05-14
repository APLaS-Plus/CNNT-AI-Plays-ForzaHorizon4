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
import cv2

# 导入混合精度训练包
from torch.cuda.amp import autocast, GradScaler

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

sys.path.append(UTILS_DIR)  # Add utils directory to path
from utils.model.lite_digit_detector import LiteDigitDetector


class DigitDataset(Dataset):
    """
    Digit Detection Dataset

    Expected dataset structure:
    - Each image size: (48, 80) or similar size, adjustable
    - Each image contains three digits at fixed positions
    - Label format: [digit1, digit2, digit3], each digit ranges from 0-9
    """

    def __init__(self, root_dir, transform=None, split="train"):
        """
        Args:
            root_dir (str): Dataset root directory
            transform (callable, optional): Image transformations
            split (str): 'train' or 'val' or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(self.root_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        # Label files should correspond to image files, format: "image_name.txt"
        # Each label file contains three digits separated by spaces
        self.labels = []
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(self.root_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    # Read three digit labels
                    digits = [int(i) for i in f.read().strip().split()]
                    if len(digits) == 3:
                        self.labels.append(digits)
                    else:
                        print(
                            f"Warning: Label file {label_file} is not in correct format, should contain 3 digits"
                        )
                        self.labels.append([0, 0, 0])  # Use default label
            else:
                print(f"Warning: Label file {label_file} not found")
                self.labels.append([0, 0, 0])  # Use default label

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # Use OpenCV to read and preprocess the image
        img = cv2.imread(img_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply median filter to remove noise
        median = cv2.medianBlur(gray, 3)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(median)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
        )

        # Morphological operations to enhance character contours
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert processed image to PIL format
        image = Image.fromarray(morph, mode="L")

        # Get label
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,  # 添加scheduler参数
    num_epochs=10,
    device="cuda",
    model_save_path=None,
    patience=15,
    min_delta=0.001,
    use_amp=True,
):
    """
    Model training function with early stopping and mixed precision
    """
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler(enabled=use_amp)

    # Move model to specified device
    model = model.to(device)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 早停计数器和最佳验证损失
    early_stop_counter = 0
    best_val_loss = float("inf")

    # 设置CUDA性能优化选项 - 修复设备类型检查
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_digits = 0
        total_digits = 0

        for images, labels in train_loader:
            # 确保数据在正确的设备上
            images = images.to(device, non_blocking=True)
            digit1_labels = labels[:, 0].to(device, non_blocking=True)
            digit2_labels = labels[:, 1].to(device, non_blocking=True)
            digit3_labels = labels[:, 2].to(device, non_blocking=True)

            # 使用混合精度训练
            with autocast(enabled=use_amp):
                # Forward pass
                digit1_pred, digit2_pred, digit3_pred = model(images)

                # Calculate loss
                loss1 = criterion(digit1_pred, digit1_labels)
                loss2 = criterion(digit2_pred, digit2_labels)
                loss3 = criterion(digit3_pred, digit3_labels)
                loss = loss1 + loss2 + loss3

            # 使用混合精度进行反向传播
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)

            # Calculate accuracy - 确保在GPU上计算
            with torch.no_grad():
                _, pred1 = torch.max(digit1_pred, 1)
                _, pred2 = torch.max(digit2_pred, 1)
                _, pred3 = torch.max(digit3_pred, 1)

                correct_digits += (pred1 == digit1_labels).sum().item()
                correct_digits += (pred2 == digit2_labels).sum().item()
                correct_digits += (pred3 == digit3_labels).sum().item()
                total_digits += labels.size(0) * 3

        # Calculate average loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_digits / total_digits

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_digits = 0
        total_digits = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                digit1_labels = labels[:, 0].to(device, non_blocking=True)
                digit2_labels = labels[:, 1].to(device, non_blocking=True)
                digit3_labels = labels[:, 2].to(device, non_blocking=True)

                # 使用混合精度进行推理
                with autocast(enabled=use_amp):
                    # Forward pass
                    digit1_pred, digit2_pred, digit3_pred = model(images)

                    # Calculate loss
                    loss1 = criterion(digit1_pred, digit1_labels)
                    loss2 = criterion(digit2_pred, digit2_labels)
                    loss3 = criterion(digit3_pred, digit3_labels)
                    loss = loss1 + loss2 + loss3

                val_loss += loss.item() * images.size(0)

                # Calculate accuracy
                _, pred1 = torch.max(digit1_pred, 1)
                _, pred2 = torch.max(digit2_pred, 1)
                _, pred3 = torch.max(digit3_pred, 1)

                correct_digits += (pred1 == digit1_labels).sum().item()
                correct_digits += (pred2 == digit2_labels).sum().item()
                correct_digits += (pred3 == digit3_labels).sum().item()
                total_digits += labels.size(0) * 3

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_digits / total_digits

        # 如果提供了调度器，使用它来调整学习率
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), str(model_save_path / "best_digit_model.pth")
            )

        # 早停机制
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"早停机制触发，训练在第 {epoch+1} 轮停止")
            break

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return history


def plot_training_history(history, model_save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(str(model_save_path / "training_history.png"))
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置CUDA优化选项
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # 预热GPU
        warming_tensor = torch.randn(8, 1, 48, 96).cuda()
        with torch.no_grad():
            for _ in range(10):
                _ = torch.mean(warming_tensor)

    # Define dataset path
    data_dir = ROOT_DIR / ".." / "dataset" / "digit"  # Modify to your dataset path
    data_dir = str(data_dir.resolve())
    model_save_path = ROOT_DIR / ".." / "run" / "light_digit_detector"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Define image transformations - updated to use the optimized image size
    transform = transforms.Compose(
        [
            transforms.Resize((48, 96)),  # Updated from (40, 120) to (48, 96)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets and data loaders
    train_dataset = DigitDataset(data_dir, transform=transform, split="train")
    val_dataset = DigitDataset(data_dir, transform=transform, split="val")

    # 优化数据加载器配置，使用pin_memory加速数据传输到GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=192,  # 增加批处理大小以利用GPU并行能力
        shuffle=True,
        num_workers=8,
        pin_memory=True,  # 使用页锁定内存加速数据传输
        prefetch_factor=2,  # 预取因子
        persistent_workers=True,  # 保持工作进程活跃
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,  # 验证时可使用更大批量
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Print dataset info
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Input data shape: {train_dataset[0][0].shape}")  # Should be (1, 48, 80)

    # Create model with optimized parameters
    model = LiteDigitDetector(input_height=48, input_width=96)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # 优化器使用更适合GPU训练的配置
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    # 学习率调度器以优化训练过程
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Check CUDA availability and select best device
    if torch.cuda.is_available():
        # 使用最快的GPU设备
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # 打印GPU信息
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
        print(
            f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("警告: 无可用GPU，使用CPU训练将会非常慢")

    # Train model with early stopping and mixed precision
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,  # 传递学习率调度器
        num_epochs=150,
        device=device,
        model_save_path=model_save_path,
        patience=30,
        min_delta=0.001,
        use_amp=True,  # 启用混合精度训练
    )

    # Plot training history
    plot_training_history(history, model_save_path)

    # Save final model
    torch.save(model.state_dict(), str(model_save_path / "final_digit_model.pth"))
    print("Training completed, model saved")
