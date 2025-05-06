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
import cv2  # Added cv2 import

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
UTILS_DIR = ROOT_DIR / ".."
UTILS_DIR = str(UTILS_DIR.resolve())

# print("Root directory:", ROOT_DIR)
# print("Utils directory:", UTILS_DIR)

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
        image = Image.fromarray(morph)

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
    num_epochs=10,
    device="cuda",
    model_save_path=None,
):
    """
    Model training function
    """
    # Move model to specified device
    model = model.to(device)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_digits = 0
        total_digits = 0

        for images, labels in train_loader:
            images = images.to(device)
            # Each label contains 3 digits
            digit1_labels = labels[:, 0].to(device)
            digit2_labels = labels[:, 1].to(device)
            digit3_labels = labels[:, 2].to(device)

            # Forward pass
            optimizer.zero_grad()
            digit1_pred, digit2_pred, digit3_pred = model(images)

            # Calculate loss
            loss1 = criterion(digit1_pred, digit1_labels)
            loss2 = criterion(digit2_pred, digit2_labels)
            loss3 = criterion(digit3_pred, digit3_labels)
            loss = loss1 + loss2 + loss3

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, pred1 = torch.max(digit1_pred, 1)
            _, pred2 = torch.max(digit2_pred, 1)
            _, pred3 = torch.max(digit3_pred, 1)

            correct_digits += (pred1 == digit1_labels).sum().item()
            correct_digits += (pred2 == digit2_labels).sum().item()
            correct_digits += (pred3 == digit3_labels).sum().item()
            total_digits += labels.size(0) * 3  # 3 digits per sample

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
                images = images.to(device)
                digit1_labels = labels[:, 0].to(device)
                digit2_labels = labels[:, 1].to(device)
                digit3_labels = labels[:, 2].to(device)

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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), str(model_save_path / "best_digit_model.pth")
            )

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

    # Define dataset path
    data_dir = ROOT_DIR / ".." / "dataset" / "digit"  # Modify to your dataset path
    data_dir = str(data_dir.resolve())
    model_save_path = ROOT_DIR / ".." / "run" / "light_digit_detector"

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Define image transformations - updated to use the optimized image size
    transform = transforms.Compose(
        [
            transforms.Resize((48, 80)),  # Updated from (40, 120) to (48, 80)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets and data loaders
    train_dataset = DigitDataset(data_dir, transform=transform, split="train")
    val_dataset = DigitDataset(data_dir, transform=transform, split="val")

    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=80, shuffle=False, num_workers=4)

    # Print dataset info
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Input data shape: {train_dataset[0][0].shape}")  # Should be (1, 48, 80)

    # Create model with optimized parameters
    model = LiteDigitDetector(input_height=48, input_width=80)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Use a slightly higher learning rate for the smaller model
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train model
    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=50,
        device=device,
        model_save_path=model_save_path,
    )

    # Plot training history
    plot_training_history(history, model_save_path)

    # Save final model
    torch.save(model.state_dict(), str(model_save_path / "final_digit_model.pth"))
    print("Training completed, model saved")
