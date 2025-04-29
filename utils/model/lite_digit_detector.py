import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    """轻量级卷积块，用于特征提取"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # 使用分组卷积来减少参数量
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
    """单个数字的分类头"""

    def __init__(self, in_features, num_classes=10):
        super(DigitClassifier, self).__init__()
        # 减小隐藏层的神经元数量
        self.fc1 = nn.Linear(in_features, 32)  # 从64减少到32
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.2)  # 减小dropout概率以提高轻量模型的表达能力

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiteDigitDetector(nn.Module):
    """轻量级数字识别模型，可以识别三个固定位置的数字"""

    def __init__(
        self, input_height=48, input_width=80
    ):
        super(LiteDigitDetector, self).__init__()
        # 输入图像预期是灰度图，如果是彩色图，会在forward中处理

        # 更轻量化的特征提取网络
        self.features = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            nn.Conv2d(32, 32, kernel_size=2)
        )

        # 计算特征提取后的特征图尺寸
        feature_height = input_height // 8 - 1
        feature_width = input_width // 8 - 1

        # 三个数字在水平方向均匀分布，我们可以将特征图划分为三部分
        self.digit_width = feature_width // 3
        self.flatten_features = (
            self.digit_width * feature_height * 32
        )  # 最后一层通道数是32而不是64

        # 打印确认特征维度
        print(
            f"Feature dimensions: {feature_height}x{feature_width}, digit_width: {self.digit_width}"
        )
        print(f"Flattened features: {self.flatten_features}")

        # 三个数字位置的分类器
        self.digit_classifier1 = DigitClassifier(self.flatten_features)
        self.digit_classifier2 = DigitClassifier(self.flatten_features)
        self.digit_classifier3 = DigitClassifier(self.flatten_features)

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (input_height, input_width)
                ),  # 保持原始比例的缩小尺寸
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def preprocess(self, image):
        """预处理输入图像"""
        if isinstance(image, torch.Tensor):
            # 如果已经是tensor，检查通道数
            if image.dim() == 4 and image.size(1) == 3:  # NCHW格式，3通道
                image = transforms.Grayscale()(image)
            elif image.dim() == 3 and image.size(0) == 3:  # CHW格式，3通道
                image = transforms.Grayscale()(image.unsqueeze(0)).squeeze(0)
            # 归一化
            if image.max() > 1.0:
                image = image / 255.0
            image = transforms.Normalize((0.5,), (0.5,))(image)
        else:
            # 如果是PIL图像或其他格式
            image = self.transform(image)

        return image

    def forward(self, x):
        """前向传播，返回三个位置的数字预测"""
        # 确保输入是灰度图
        if x.dim() == 4 and x.size(1) == 3:  # Batch of RGB images
            x = torch.mean(x, dim=1, keepdim=True)  # 简单地转为灰度
        elif x.dim() == 3 and x.size(0) == 3:  # Single RGB image
            x = torch.mean(x, dim=0, keepdim=True).unsqueeze(0)

        # 特征提取
        features = self.features(x)
        batch_size = features.size(0)

        # 根据位置分割特征图
        digit1_features = features[:, :, :, : self.digit_width].reshape(batch_size, -1)
        digit2_features = features[
            :, :, :, self.digit_width : 2 * self.digit_width
        ].reshape(batch_size, -1)
        digit3_features = features[:, :, :, 2 * self.digit_width :].reshape(
            batch_size, -1
        )

        # 对每个位置的数字进行分类
        digit1_pred = self.digit_classifier1(digit1_features)
        digit2_pred = self.digit_classifier2(digit2_features)
        digit3_pred = self.digit_classifier3(digit3_features)

        return digit1_pred, digit2_pred, digit3_pred

    def predict(self, image):
        """预测图像中三个位置的数字"""
        # 预处理
        processed_image = self.preprocess(image)
        if processed_image.dim() == 3:  # 单张图片
            processed_image = processed_image.unsqueeze(0)

        # 推理
        self.eval()
        with torch.no_grad():
            digit1_logits, digit2_logits, digit3_logits = self(processed_image)

        # 获取预测结果
        digit1 = digit1_logits.argmax(dim=1)
        digit2 = digit2_logits.argmax(dim=1)
        digit3 = digit3_logits.argmax(dim=1)

        return digit1.item(), digit2.item(), digit3.item()

    def save_weights(self, path):
        """保存模型权重"""
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """加载模型权重"""
        self.load_state_dict(torch.load(path))
