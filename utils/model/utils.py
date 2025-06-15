"""
Thanks the code from Ultralytics YOLO11 repository.
Utility modules for neural network components including convolutional layers,
attention mechanisms, and activation functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    Automatically calculate padding to maintain 'same' shape outputs.

    Args:
        k (int | list): Kernel size(s)
        p (int | list, optional): Padding size(s). If None, auto-calculate
        d (int): Dilation factor

    Returns:
        int | list: Calculated padding size(s)
    """
    if d > 1:
        # Calculate actual kernel size with dilation
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        # Auto-calculate padding for 'same' output
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class Bottleneck(nn.Module):
    """
    Bottleneck residual block with squeeze-and-expand architecture.

    This module implements a standard bottleneck block commonly used in ResNet-like
    architectures. It first reduces the channel dimension, applies 3x3 convolution,
    then expands back to the original dimension with a residual connection.

    Attributes:
        cv1 (Conv): 1x1 convolution for channel reduction
        cv2 (Conv): 3x3 convolution for feature extraction
        cv3 (Conv): 1x1 convolution for channel expansion (no activation)
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Bottleneck residual block with given parameters.

        Args:
            c1 (int): Number of input channels
            c2 (int): Number of output channels
            k (int): Kernel size for first and last convolutions
            s (int): Stride for convolutions
            p (int, optional): Padding size
            g (int): Number of groups for convolutions
            d (int): Dilation factor
            act (bool | nn.Module): Activation function configuration
        """
        super().__init__()
        # First 1x1 conv: reduce channels by half
        self.cv1 = Conv(c1, c2 // 2, k, s, p, g, d, act)
        # 3x3 conv: maintain reduced channel count
        self.cv2 = Conv(c2 // 2, c2 // 2, 3, 1, p, g, d, act)
        # Final 1x1 conv: expand to output channels (no activation for residual)
        self.cv3 = Conv(c2 // 2, c2, k, s, p, g, d, act=False)

    def forward(self, x):
        """
        Apply bottleneck transformation with residual connection.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        # Apply bottleneck transformation and add residual connection
        return self.cv3(self.cv2(self.cv1(x))) + x


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM) - A parameter-free attention mechanism.

    SimAM computes attention weights based on the energy function that measures
    the linear separability between each neuron and all other neurons in the
    same channel. This approach requires no additional parameters.

    Reference:
        SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
        https://proceedings.mlr.press/v139/yang21o.html

    Attributes:
        e_lambda (float): Energy function regularization parameter
        eps (float): Small constant to avoid division by zero
    """

    def __init__(self, e_lambda=1e-4, eps=1e-5):
        """
        Initialize SimAM attention module.

        Args:
            e_lambda (float): Regularization parameter for energy function
            eps (float): Small epsilon value to prevent division by zero
        """
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda
        self.eps = eps

    def forward(self, x):
        """
        Apply SimAM attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Attention-weighted output tensor
        """
        # Get tensor dimensions: Batch, Channel, Height, Width
        B, C, H, W = x.size()
        n = H * W  # Total spatial dimensions

        # Calculate mean and variance for each channel separately
        mean = x.mean(dim=[2, 3], keepdim=True)  # Channel-wise mean: (B, C, 1, 1)
        var = ((x - mean) ** 2).mean(
            dim=[2, 3], keepdim=True
        )  # Channel-wise variance: (B, C, 1, 1)

        # Compute energy function for attention weight calculation
        # Energy measures how distinguishable each pixel is from the channel mean
        e_lambda = self.e_lambda
        energy = ((x - mean) ** 2) / (var + self.eps) + e_lambda * x

        # Convert energy to attention weights using sigmoid activation
        attention = torch.sigmoid(-energy)

        # Apply attention weights to input
        return x * attention
