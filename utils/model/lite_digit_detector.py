import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    """Lightweight convolutional block for feature extraction"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # 使用融合操作来优化GPU效率
        self.fused_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=1 if in_channels < 4 else 2,
                bias=False,  # 与批归一化配合使用时不需要偏置
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.fused_conv(x)


class DigitClassifier(nn.Module):
    """Classification head for a single digit"""

    def __init__(self, in_features, num_classes=10):
        super(DigitClassifier, self).__init__()
        # 优化分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class LiteDigitDetector(nn.Module):
    """Lightweight digit recognition model capable of recognizing three fixed-position digits"""

    def __init__(self, input_height=48, input_width=96):
        super(LiteDigitDetector, self).__init__()
        # Input image is expected to be grayscale; if it's colored, it will be processed in forward

        # 更高效的特征提取网络
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
        )

        # Calculate feature map dimensions after feature extraction
        feature_height = input_height // 16
        feature_width = input_width // 16

        # Three digits are evenly distributed horizontally; we can divide the feature map into three parts
        self.digit_width = feature_width // 3
        self.flatten_features = self.digit_width * feature_height * 64
        # Print to confirm feature dimensions
        print(
            f"Feature dimensions: {feature_height}x{feature_width}, digit_width: {self.digit_width}"
        )
        print(f"Flattened features: {self.flatten_features}")

        # Use a single shared classifier for all three digit positions
        self.digit_classifier = DigitClassifier(self.flatten_features)

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
        # 确保图像预处理在GPU上进行，如果可能
        if isinstance(image, torch.Tensor):
            device = image.device  # 获取输入设备
            # If already a tensor, check the number of channels
            if image.dim() == 4 and image.size(1) == 3:  # NCHW format, 3 channels
                image = transforms.Grayscale()(image)
            elif image.dim() == 3 and image.size(0) == 3:  # CHW format, 3 channels
                image = transforms.Grayscale()(image.unsqueeze(0)).squeeze(0)
            # Normalize
            if image.max() > 1.0:
                image = image / 255.0
            # 归一化操作更好地在GPU上完成
            if image.dim() == 4:  # 批处理
                image = transforms.Normalize((0.5,), (0.5,))(image)
            else:  # 单张图像
                image = transforms.Normalize((0.5,), (0.5,))(
                    image.unsqueeze(0)
                ).squeeze(0)
        else:
            # If it's a PIL image or other format
            image = self.transform(image)

        return image

    def forward(self, x):
        """Forward pass, returns predictions for three digit positions"""
        # 确保输入是灰度图像，并在GPU上进行操作
        if x.dim() == 4 and x.size(1) == 3:  # 批处理的RGB图像
            x = torch.mean(x, dim=1, keepdim=True)  # 在GPU上转为灰度
        elif x.dim() == 3 and x.size(0) == 3:  # Single RGB image
            x = torch.mean(x, dim=0, keepdim=True).unsqueeze(0)
        # print(f"Input image size: {x.size()}")
        # Feature extraction
        features = self.features(x)
        batch_size = features.size(0)
        # print(f"Feature map size: {features.size()}")
        # Split feature map by position
        digit1_features = features[:, :, :, : self.digit_width].reshape(batch_size, -1)
        digit2_features = features[
            :, :, :, self.digit_width : 2 * self.digit_width
        ].reshape(batch_size, -1)
        digit3_features = features[:, :, :, 2 * self.digit_width :].reshape(
            batch_size, -1
        )

        # 合并特征进行批处理，提高GPU利用率
        combined_features = torch.cat(
            [digit1_features, digit2_features, digit3_features], dim=0
        )
        combined_preds = self.digit_classifier(combined_features)

        # 分割结果
        digit1_pred, digit2_pred, digit3_pred = torch.split(combined_preds, batch_size)

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

    def load_weights(self, path, map_location=None):
        """Load model weights with device mapping support"""
        if map_location is None and torch.cuda.is_available():
            map_location = torch.device("cuda")
        self.load_state_dict(torch.load(path, map_location=map_location))
