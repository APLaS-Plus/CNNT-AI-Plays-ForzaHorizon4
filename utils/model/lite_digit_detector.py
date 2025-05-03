import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    """Lightweight convolutional block for feature extraction"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # Use grouped convolution to reduce the number of parameters
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=1 if in_channels < 4 else 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DigitClassifier(nn.Module):
    """Classification head for a single digit"""

    def __init__(self, in_features, num_classes=10):
        super(DigitClassifier, self).__init__()
        # Reduce hidden layer neurons to optimize model size
        self.fc1 = nn.Linear(in_features, 32)  # Reduced from 64 to 32
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)  # Lower dropout rate to improve lightweight model's expressiveness

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiteDigitDetector(nn.Module):
    """Lightweight digit recognition model capable of recognizing three fixed-position digits"""

    def __init__(
        self, input_height=48, input_width=80
    ):
        super(LiteDigitDetector, self).__init__()
        # Input image is expected to be grayscale; if it's colored, it will be processed in forward

        # More lightweight feature extraction network
        self.features = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            nn.Conv2d(32, 32, kernel_size=2)
        )

        # Calculate feature map dimensions after feature extraction
        feature_height = input_height // 8 - 1
        feature_width = input_width // 8 - 1

        # Three digits are evenly distributed horizontally; we can divide the feature map into three parts
        self.digit_width = feature_width // 3
        self.flatten_features = (
            self.digit_width * feature_height * 32
        )  # The last layer channel count is 32 instead of 64

        # Print to confirm feature dimensions
        print(
            f"Feature dimensions: {feature_height}x{feature_width}, digit_width: {self.digit_width}"
        )
        print(f"Flattened features: {self.flatten_features}")

        # Classifiers for three digit positions
        self.digit_classifier1 = DigitClassifier(self.flatten_features)
        self.digit_classifier2 = DigitClassifier(self.flatten_features)
        self.digit_classifier3 = DigitClassifier(self.flatten_features)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (input_height, input_width)
                ),  # Resize while maintaining original aspect ratio
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def preprocess(self, image):
        """Preprocess input image"""
        if isinstance(image, torch.Tensor):
            # If already a tensor, check the number of channels
            if image.dim() == 4 and image.size(1) == 3:  # NCHW format, 3 channels
                image = transforms.Grayscale()(image)
            elif image.dim() == 3 and image.size(0) == 3:  # CHW format, 3 channels
                image = transforms.Grayscale()(image.unsqueeze(0)).squeeze(0)
            # Normalize
            if image.max() > 1.0:
                image = image / 255.0
            image = transforms.Normalize((0.5,), (0.5,))(image)
        else:
            # If it's a PIL image or other format
            image = self.transform(image)

        return image

    def forward(self, x):
        """Forward pass, returns predictions for three digit positions"""
        # Ensure input is grayscale
        if x.dim() == 4 and x.size(1) == 3:  # Batch of RGB images
            x = torch.mean(x, dim=1, keepdim=True)  # Convert to grayscale
        elif x.dim() == 3 and x.size(0) == 3:  # Single RGB image
            x = torch.mean(x, dim=0, keepdim=True).unsqueeze(0)

        # Feature extraction
        features = self.features(x)
        batch_size = features.size(0)

        # Split feature map by position
        digit1_features = features[:, :, :, : self.digit_width].reshape(batch_size, -1)
        digit2_features = features[
            :, :, :, self.digit_width : 2 * self.digit_width
        ].reshape(batch_size, -1)
        digit3_features = features[:, :, :, 2 * self.digit_width :].reshape(
            batch_size, -1
        )

        # Classify digits at each position
        digit1_pred = self.digit_classifier1(digit1_features)
        digit2_pred = self.digit_classifier2(digit2_features)
        digit3_pred = self.digit_classifier3(digit3_features)

        return digit1_pred, digit2_pred, digit3_pred

    def predict(self, image):
        """Predict three digits in the image"""
        # Preprocess
        processed_image = self.preprocess(image)
        if processed_image.dim() == 3:  # Single image
            processed_image = processed_image.unsqueeze(0)

        # Inference
        self.eval()
        with torch.no_grad():
            digit1_logits, digit2_logits, digit3_logits = self(processed_image)

        # Get prediction results
        digit1 = digit1_logits.argmax(dim=1)
        digit2 = digit2_logits.argmax(dim=1)
        digit3 = digit3_logits.argmax(dim=1)

        return digit1.item(), digit2.item(), digit3.item()

    def save_weights(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """Load model weights"""
        self.load_state_dict(torch.load(path))
