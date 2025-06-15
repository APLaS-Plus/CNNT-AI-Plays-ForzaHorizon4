import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .utils import GhostConv, CBAM, SimAM, Bottleneck


class ConvBlock(nn.Module):
    """
    Lightweight convolutional block optimized for efficient feature extraction.

    This block combines convolution, batch normalization, activation, and pooling
    operations in a fused manner to improve GPU efficiency and reduce memory overhead.

    Attributes:
        fused_conv (nn.Sequential): Fused convolution operations for efficiency
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Initialize ConvBlock with specified parameters.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolutional kernel
            stride (int): Stride for convolution
            padding (int): Padding for convolution
        """
        super(ConvBlock, self).__init__()
        # Use fused operations to optimize GPU efficiency
        self.fused_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=1,
                bias=False,  # No bias needed when using batch normalization
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),  # Swish activation function
            nn.MaxPool2d(2),  # 2x2 max pooling for downsampling
        )

    def forward(self, x):
        """Apply fused convolution operations to input tensor."""
        return self.fused_conv(x)


class DigitClassifier(nn.Module):
    """
    Classification head for single digit recognition (0-9).

    This module implements a simple but effective classifier with dropout
    regularization to prevent overfitting on digit classification tasks.

    Attributes:
        classifier (nn.Sequential): Multi-layer perceptron for classification
    """

    def __init__(self, in_features, num_classes=10):
        """
        Initialize digit classifier.

        Args:
            in_features (int): Number of input features
            num_classes (int): Number of output classes (default: 10 for digits 0-9)
        """
        super(DigitClassifier, self).__init__()
        # Optimized classifier structure with expansion and contraction
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features * 2),  # Feature expansion
            nn.SiLU(inplace=True),  # Swish activation
            nn.Dropout(0.1),  # Regularization
            nn.Linear(in_features * 2, num_classes),  # Final classification layer
        )

    def forward(self, x):
        """
        Classify input features into digit classes.

        Args:
            x (torch.Tensor): Input feature tensor

        Returns:
            torch.Tensor: Classification logits for each digit class
        """
        return self.classifier(x)


class LiteDigitDetector(nn.Module):
    """
    Lightweight digit recognition model for detecting three fixed-position digits.

    This model is specifically designed for speed recognition in racing games,
    where three digits appear at fixed horizontal positions. It uses efficient
    convolutions, attention mechanisms, and shared classifiers to minimize
    computational overhead while maintaining accuracy.

    Architecture:
    - Feature extraction using GhostConv, SimAM attention, and bottleneck blocks
    - Spatial division for three digit positions
    - Shared classifier for all three positions

    Attributes:
        features (nn.Sequential): Feature extraction backbone
        digit_classifier (DigitClassifier): Shared classifier for all positions
        transform (transforms.Compose): Image preprocessing pipeline
    """

    def __init__(self, input_height=48, input_width=96):
        """
        Initialize LiteDigitDetector with specified input dimensions.

        Args:
            input_height (int): Expected input image height
            input_width (int): Expected input image width
        """
        super(LiteDigitDetector, self).__init__()
        # Input image is expected to be grayscale; RGB images will be converted in forward pass

        # Efficient feature extraction network using modern components
        self.features = nn.Sequential(
            GhostConv(1, 16),  # Efficient convolution with ghost features
            nn.Dropout(0.1),  # Light regularization
            SimAM(),  # Parameter-free attention mechanism
            nn.MaxPool2d(2),  # Spatial downsampling
            GhostConv(16, 32),  # Expand channels efficiently
            nn.Dropout(0.1),
            SimAM(),  # Another attention layer
            nn.MaxPool2d(2),
            Bottleneck(32, 32),  # Residual bottleneck block
            nn.Dropout(0.1),
            ConvBlock(32, 64),  # Standard convolution block
            nn.Dropout(0.1),
            Bottleneck(64, 64),  # Another residual block
            nn.Dropout(0.1),
            CBAM(64, kernel_size=7),  # Convolutional Block Attention Module
            nn.Conv2d(64, 64, 2, 2, 0),  # Final downsampling
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Calculate feature map dimensions after feature extraction (16x downsampling)
        feature_height = input_height // 16
        feature_width = input_width // 16

        # Three digits are evenly distributed horizontally; divide feature map into three parts
        self.digit_width = feature_width // 3
        self.flatten_features = self.digit_width * feature_height * 64

        # Print feature dimensions for debugging and verification
        print(
            f"Feature dimensions: {feature_height}x{feature_width}, digit_width: {self.digit_width}"
        )
        print(f"Flattened features: {self.flatten_features}")

        # Use a single shared classifier for all three digit positions to reduce parameters
        self.digit_classifier = DigitClassifier(self.flatten_features)

        # Image preprocessing pipeline for consistent input format
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),  # Convert to grayscale
                transforms.Resize(
                    (input_height, input_width)
                ),  # Resize while maintaining aspect ratio
                transforms.ToTensor(),  # Convert to tensor [0,1]
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
            ]
        )

    def preprocess(self, image):
        """
        Preprocess input image for model inference.

        Handles both PIL images and torch tensors, ensuring consistent
        format and normalization. Operations are performed on GPU when possible.

        Args:
            image: Input image (PIL Image or torch.Tensor)

        Returns:
            torch.Tensor: Preprocessed image tensor ready for inference
        """
        # Ensure image preprocessing is performed on GPU when possible
        if isinstance(image, torch.Tensor):
            device = image.device  # Get input device for consistency

            # Handle RGB to grayscale conversion for tensor inputs
            if (
                image.dim() == 4 and image.size(1) == 3
            ):  # Batch of RGB images (NCHW format)
                image = transforms.Grayscale()(image)
            elif (
                image.dim() == 3 and image.size(0) == 3
            ):  # Single RGB image (CHW format)
                image = transforms.Grayscale()(image.unsqueeze(0)).squeeze(0)

            # Normalize pixel values to [0,1] range if needed
            if image.max() > 1.0:
                image = image / 255.0

            # Apply normalization to [-1,1] range for better training stability
            if image.dim() == 4:  # Batch processing
                image = transforms.Normalize((0.5,), (0.5,))(image)
            else:  # Single image processing
                image = transforms.Normalize((0.5,), (0.5,))(
                    image.unsqueeze(0)
                ).squeeze(0)
        else:
            # Handle PIL images or other formats using standard transform pipeline
            image = self.transform(image)

        return image

    def forward(self, x):
        """
        Forward pass for three-digit recognition.

        Processes input image through feature extraction, spatially divides
        the feature map for three digit positions, and applies shared
        classification to each position.

        Args:
            x (torch.Tensor): Input image tensor

        Returns:
            tuple: (digit1_pred, digit2_pred, digit3_pred) - Logits for each position
        """
        # Ensure input is grayscale and perform GPU-optimized conversion if needed
        if x.dim() == 4 and x.size(1) == 3:  # Batch of RGB images
            x = torch.mean(x, dim=1, keepdim=True)  # Convert to grayscale on GPU
        elif x.dim() == 3 and x.size(0) == 3:  # Single RGB image
            x = torch.mean(x, dim=0, keepdim=True).unsqueeze(0)

        # Extract features using the backbone network
        features = self.features(x)
        batch_size = features.size(0)

        # Split feature map by horizontal position for three digits
        # Each digit occupies 1/3 of the feature width
        digit1_features = features[:, :, :, : self.digit_width].reshape(batch_size, -1)
        digit2_features = features[
            :, :, :, self.digit_width : 2 * self.digit_width
        ].reshape(batch_size, -1)
        digit3_features = features[:, :, :, 2 * self.digit_width :].reshape(
            batch_size, -1
        )

        # Combine features for batch processing to improve GPU utilization
        combined_features = torch.cat(
            [digit1_features, digit2_features, digit3_features], dim=0
        )
        combined_preds = self.digit_classifier(combined_features)

        # Split results back to individual digit predictions
        digit1_pred, digit2_pred, digit3_pred = torch.split(combined_preds, batch_size)

        return digit1_pred, digit2_pred, digit3_pred

    def predict(self, image):
        """
        Predict three digits in the input image.

        This method handles the complete inference pipeline including
        preprocessing, model forward pass, and post-processing to return
        the predicted digit values.

        Args:
            image: Input image (PIL Image or torch.Tensor)

        Returns:
            tuple: (digit1, digit2, digit3) - Predicted digit values as integers
        """
        # Preprocess input image
        processed_image = self.preprocess(image)
        if processed_image.dim() == 3:  # Add batch dimension for single image
            processed_image = processed_image.unsqueeze(0)

        # Perform inference in evaluation mode
        self.eval()
        with torch.no_grad():
            digit1_logits, digit2_logits, digit3_logits = self(processed_image)

        # Get prediction results by taking argmax of logits
        digit1 = digit1_logits.argmax(dim=1)
        digit2 = digit2_logits.argmax(dim=1)
        digit3 = digit3_logits.argmax(dim=1)

        return digit1.item(), digit2.item(), digit3.item()

    def save_weights(self, path):
        """
        Save model weights to specified path.

        Args:
            path (str): File path to save the model weights
        """
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        """
        Load model weights from specified path with device mapping support.

        Args:
            path (str): File path to load the model weights from
            map_location: Device mapping specification for loading weights
        """
        if map_location is None and torch.cuda.is_available():
            map_location = torch.device("cuda")
        self.load_state_dict(torch.load(path, map_location=map_location))
