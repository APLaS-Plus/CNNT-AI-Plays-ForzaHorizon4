import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Conv, CBAM, SimAM, Bottleneck, GhostConv


class Conv2dBlock(nn.Module):
    """
    2D Convolutional block with batch normalization, activation, and downsampling.

    This block combines standard convolution operations with regularization
    and spatial downsampling for efficient feature extraction in the CNN backbone.

    Attributes:
        op (nn.Sequential): Sequential operations including conv, norm, activation, and pooling
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Initialize Conv2dBlock with specified parameters.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of convolutional kernel
            stride (int): Stride for first convolution
            padding (int): Padding for first convolution
        """
        super(Conv2dBlock, self).__init__()
        self.op = nn.Sequential(
            # First convolution with specified parameters
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),  # Batch normalization for training stability
            nn.SiLU(inplace=True),  # Swish activation function
            # Second convolution for downsampling (2x2 kernel, stride 2)
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1),  # Light regularization
        )

    def forward(self, x):
        """Apply conv2d block operations to input tensor."""
        x = self.op(x)
        return x


class SingleEncoder(nn.Module):
    """
    Encoder for single scalar values (e.g., speed) to create sequence embeddings.

    This module transforms a single scalar input into a sequence of embedded vectors,
    allowing scalar values to be processed alongside sequence data in transformers.

    Attributes:
        seq_len (int): Target sequence length for output
        embed_dim (int): Embedding dimension for each sequence element
        fill_seq (nn.Linear): Linear layer to expand scalar to sequence
        embedding (nn.Sequential): Multi-layer embedding network
    """

    def __init__(self, seq_len, embed_dim):
        """
        Initialize SingleEncoder with sequence and embedding dimensions.

        Args:
            seq_len (int): Length of output sequence
            embed_dim (int): Dimension of embedding vectors
        """
        super(SingleEncoder, self).__init__()
        self.seq_len = seq_len
        self.fill_seq = nn.Linear(1, seq_len)  # Expand scalar to sequence
        self.embed_dim = embed_dim
        # Multi-layer embedding network for rich representations
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
        )

    def forward(self, x):
        """
        Transform scalar input to embedded sequence.

        Args:
            x (torch.Tensor): Input scalar tensor

        Returns:
            torch.Tensor: Embedded sequence tensor of shape (batch, seq_len, embed_dim)
        """
        x = x.unsqueeze(-1)  # Add feature dimension: (batch,) -> (batch, 1)
        x = self.fill_seq(x)  # Expand to sequence: (batch, 1) -> (batch, seq_len)
        x = x.unsqueeze(
            -1
        )  # Add embedding input dim: (batch, seq_len) -> (batch, seq_len, 1)
        x = self.embedding(
            x
        )  # Apply embedding: (batch, seq_len, 1) -> (batch, seq_len, embed_dim)
        return x


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) for controlling information flow.

    GLU applies a gating mechanism to control which information passes through,
    using a sigmoid gate to modulate the linear transformation output.

    Reference:
        Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)

    Attributes:
        linear (nn.Linear): Main linear transformation
        gate (nn.Linear): Gate computation for information control
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize GLU with input and output dimensions.

        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
        """
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Main transformation
        self.gate = nn.Linear(input_dim, output_dim)  # Gate computation

    def forward(self, x):
        """
        Apply gated linear unit transformation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Gated output tensor
        """
        # Element-wise multiplication of linear output and sigmoid gate
        return self.linear(x) * torch.sigmoid(self.gate(x))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.

    SE blocks adaptively recalibrate channel-wise feature responses by
    explicitly modeling interdependencies between channels through a
    squeeze-and-excitation mechanism.

    Reference:
        Squeeze-and-Excitation Networks (Hu et al., 2018)

    Attributes:
        fc (nn.Sequential): Fully connected layers for channel attention computation
    """

    def __init__(self, channel, reduction=16):
        """
        Initialize SE block with channel attention mechanism.

        Args:
            channel (int): Number of input channels
            reduction (int): Channel reduction ratio for bottleneck
        """
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # Channel squeeze
            nn.ReLU(),  # Non-linearity
            nn.Linear(channel // reduction, channel),  # Channel excitation
            nn.Sigmoid(),  # Attention weights
        )

    def forward(self, x):
        """
        Apply squeeze-and-excitation channel attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, channel)

        Returns:
            torch.Tensor: Channel-attended output tensor
        """
        # Global average pooling across sequence dimension for channel statistics
        w = self.fc(x.mean(dim=1))  # (batch, channel)
        # Apply channel attention weights
        return x * w.unsqueeze(1)  # Broadcast to (batch, 1, channel)


class Feature_extractor(nn.Module):
    """
    CNN-based feature extractor for transforming images into sequence embeddings.

    This module processes input images through a series of convolutional layers,
    attention mechanisms, and bottleneck blocks to extract rich spatial features
    that are then reshaped into sequence format for transformer processing.

    Architecture progression:
    960x640x3 -> ... -> 3x2x512 -> embedded sequence of 6 tokens with embedding_dim features

    Attributes:
        conv (nn.Sequential): Convolutional feature extraction backbone
        embedding (nn.Sequential): Feature embedding layers
    """

    def __init__(self, input_w=960, input_h=640, embedding_dim=1024):
        """
        Initialize feature extractor with input dimensions and embedding size.

        Args:
            input_w (int): Input image width
            input_h (int): Input image height
            embedding_dim (int): Output embedding dimension per spatial location
        """
        super(Feature_extractor, self).__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.embedding_dim = embedding_dim

        # Calculate expected final spatial dimensions after all downsampling
        self.expected_final_h = 2
        self.expected_final_w = 3

        # Progressive feature extraction with attention and residual connections
        self.conv = nn.Sequential(
            # Initial feature extraction: 960x640x3 -> 480x320x16
            GhostConv(3, 16),  # Efficient convolution with ghost features
            nn.MaxPool2d(2),  # Spatial downsampling
            nn.Dropout(0.1),  # Light regularization
            SimAM(),  # Parameter-free spatial attention
            # First downsampling stage: 480x320x16 -> 120x80x32
            Conv2dBlock(16, 32, 5, 2, 2),  # 5x5 conv with stride 2
            Bottleneck(32, 32),  # Residual bottleneck block
            CBAM(32),  # Channel and spatial attention
            # Second downsampling stage: 120x80x32 -> 60x40x64
            Conv2dBlock(32, 64),  # Standard conv block with 2x downsampling
            Bottleneck(64, 64),  # Another residual block
            CBAM(64),  # Attention mechanism
            # Third downsampling stage: 60x40x64 -> 30x20x128
            Conv2dBlock(64, 128),
            Bottleneck(128, 128),
            CBAM(128),
            # Fourth downsampling stage: 30x20x128 -> 15x10x256
            Conv2dBlock(128, 256),
            Bottleneck(256, 256),
            CBAM(256),
            # Final feature extraction: 15x10x256 -> 3x2x512
            Conv(256, 512, 5, 5, 0),  # Large kernel convolution for final downsampling
            Bottleneck(512, 512),  # Final residual processing
            CBAM(512),  # Final attention mechanism
            nn.Dropout(0.1),  # Final regularization
        )

        # Embedding layers to transform features to desired embedding dimension
        self.embedding = nn.Sequential(
            nn.Linear(512, self.embedding_dim),  # First embedding layer
            nn.SiLU(inplace=True),  # Swish activation
            nn.Linear(self.embedding_dim, self.embedding_dim),  # Second embedding layer
            nn.SiLU(inplace=True),  # Final activation
        )

    def forward(self, x):
        """
        Extract features from input image and convert to sequence embedding.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, input_h, input_w)

        Returns:
            torch.Tensor: Sequence embeddings of shape (batch, seq_len, embedding_dim)
                         where seq_len = final_h * final_w
        """
        # Validate input dimensions to ensure compatibility
        if x.size(-2) != self.input_h or x.size(-1) != self.input_w:
            raise ValueError(
                f"Expected input size ({self.input_h}, {self.input_w}), got ({x.size(-2)}, {x.size(-1)})"
            )

        # Apply convolutional feature extraction
        x = self.conv(x)
        batch_size, channels, height, width = x.size()

        # Verify output spatial dimensions match expectations
        if height != self.expected_final_h or width != self.expected_final_w:
            print(
                f"Warning: Expected final size ({self.expected_final_h}, {self.expected_final_w}), got ({height}, {width})"
            )

        # Reshape spatial features to sequence format: (batch, channels, h, w) -> (batch, h*w, channels)
        feature = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # Apply embedding transformation to get final sequence embeddings
        output = self.embedding(feature)
        return output


class VitImageEncoder(nn.Module):
    def __init__(self, patch_w, patch_h, embedding_dim):
        super(VitImageEncoder, self).__init__()
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.embedding_dim = embedding_dim
        self.num_patches = int(patch_h * patch_w)
        self.learned_positional_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embedding_dim)
        )

    def forward(self, x):
        batch_size, patch_size, channels = x.size()

        # 验证输入维度
        if channels != self.embedding_dim:
            raise ValueError(
                f"Expected input channels {self.embedding_dim}, got {channels}"
            )

        # 验证patch数量
        if patch_size != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {patch_size}")

        return x + self.learned_positional_embedding


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=0.1
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x, attn_mask=None)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class CNNViTBlock(nn.Module):

    def __init__(
        self,
        input_w=960,
        input_h=640,
        embedding_dim=1024,
        num_of_simgle_layers=2,
        num_heads=8,
    ):
        super(CNNViTBlock, self).__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.embedding_dim = embedding_dim

        if self.embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim1 ({self.embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Feature extractor
        self.feature_extractor = Feature_extractor(input_w, input_h, self.embedding_dim)

        # Image encoder
        patch_w = 3
        patch_h = 2
        self.image_encoder = VitImageEncoder(patch_w, patch_h, self.embedding_dim)

        # Single encoder for speed, 1 -> 2, embedding_dim1
        self.speed_encoder = SingleEncoder(2, self.embedding_dim)

        # Transformer blocks1
        self.transformer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embedding_dim, num_heads),
                    GLU(self.embedding_dim, self.embedding_dim),
                )
                for _ in range(num_of_simgle_layers)
            ]
        )

        # Downsample1: 8, embedding_dim -> 4, embedding_dim//2
        self.downsample1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 4, 4)
        self.downsample2 = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.GELU(),
        )

    def forward(self, img, speed):
        img_features = self.image_encoder(self.feature_extractor(img))
        speed_tokens = self.speed_encoder(speed)

        combined_token = torch.cat([img_features, speed_tokens], dim=1)

        output = combined_token
        for block in self.transformer_blocks:
            output = block(output)

        output = output.transpose(1, 2)
        output = self.downsample1(output)
        output = output.transpose(1, 2)
        output = self.downsample2(output)

        return output


class ResDownsample(nn.Module):
    def __init__(self, embed_dim, kernel_size=5, stride=1, padding=2):
        super(ResDownsample, self).__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv1d(
            self.embed_dim, self.embed_dim, kernel_size, stride, padding
        )
        self.norm = nn.BatchNorm1d(self.embed_dim)
        self.activation = nn.GELU()
        self.downsample = nn.Conv1d(
            self.embed_dim, self.embed_dim, kernel_size=2, stride=2
        )

    def forward(self, x):
        if x.size(1) != self.embed_dim:
            raise ValueError(
                f"Expected input channels {self.embed_dim}, got {x.size(1)}"
            )
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x + residual
        x = self.downsample(x)
        return x


class CNN_Transformer(nn.Module):
    """
    Combined CNN-Transformer model for autonomous driving control prediction.

    This model integrates computer vision (CNN) with sequential modeling (Transformer)
    to predict steering and acceleration controls for autonomous driving. It processes
    image sequences along with speed information to make driving decisions.

    Architecture:
    1. CNN-ViT block processes current image and speed
    2. Temporal transformer processes sequence of features over time
    3. Separate branches for acceleration and steering prediction
    4. Combined output layer for final control signals

    Attributes:
        cnnvit (CNNViTBlock): Combined CNN and Vision Transformer for current frame processing
        timepositional_embedding (nn.Parameter): Learnable positional embeddings for temporal sequence
        timetransformer (nn.ModuleList): Stack of transformer blocks for temporal modeling
        downsample (ResDownsample): Residual downsampling for sequence compression
        acc_branch1/2 (nn.Sequential): Acceleration prediction branches
        steering_branch1/2 (nn.Sequential): Steering prediction branches
        combined_branch (nn.Sequential): Feature combination layer
        acc_outputlayer (nn.Sequential): Final acceleration output with tanh activation
        steering_outputlayer (nn.Sequential): Final steering output with tanh activation
    """

    def __init__(
        self,
        input_len=2,  # Number of features per time step
        seq_len=64,  # Sequence length for temporal modeling
        embed_dim=256,  # Embedding dimension
        num_heads=8,  # Number of attention heads
        num_layers=2,  # Number of transformer layers
        batch_size=64,  # Batch size (affects processing mode)
    ):
        """
        Initialize CNN-Transformer model with specified architecture parameters.

        Args:
            input_len (int): Number of features per time step (image features + speed)
            seq_len (int): Length of temporal sequence for transformer processing
            embed_dim (int): Embedding dimension for transformer
            num_heads (int): Number of attention heads in transformer
            num_layers (int): Number of transformer layers
            batch_size (int): Batch size (1 for inference mode, >1 for training mode)
        """
        super(CNN_Transformer, self).__init__()
        self.input_len = input_len
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokens_len = self.seq_len * self.input_len  # Total tokens: 64*2 = 128
        self.batch_size = batch_size

        # CNN-ViT block for processing current image and speed
        self.cnnvit = CNNViTBlock()  # Output shape: (batch, 2, 256)

        # Learnable temporal positional embeddings for 128 tokens with 256 dimensions
        self.timepositional_embedding = nn.Parameter(
            torch.randn(1, self.tokens_len, self.embed_dim)
        )

        # Stack of transformer blocks for temporal sequence modeling
        self.timetransformer = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embed_dim, self.num_heads),
                    GLU(
                        self.embed_dim, self.embed_dim
                    ),  # Gated linear unit for information control
                )
                for _ in range(self.num_layers)
            ]
        )

        # Residual downsampling to compress sequence length
        self.downsample = ResDownsample(self.embed_dim)

        # Acceleration prediction branches
        self.acc_branch1 = nn.Sequential(
            nn.Linear(
                self.tokens_len // 2, self.tokens_len
            ),  # Sequence dimension expansion
            nn.GELU(),
            nn.Linear(self.tokens_len, 1),  # Sequence compression to single value
        )
        self.acc_branch2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),  # Feature expansion
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim // 2),  # Feature compression
        )

        # Steering prediction branches (parallel to acceleration)
        self.steering_branch1 = nn.Sequential(
            nn.Linear(self.tokens_len // 2, self.tokens_len),
            nn.GELU(),
            nn.Linear(self.tokens_len, 1),
        )
        self.steering_branch2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim // 2),
        )

        # Combined feature processing for final predictions
        self.combined_branch = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        # Final output layers with tanh activation for bounded control signals
        self.acc_outputlayer = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Tanh(),  # Output range: [-1, 1]
        )

        self.steering_outputlayer = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Tanh(),  # Output range: [-1, 1]
        )

    def forward(self, new_img, new_speed, feature_queue):
        """
        Forward pass for control prediction.

        Processes current image and speed information, maintains temporal context,
        and predicts acceleration and steering controls.

        Args:
            new_img (torch.Tensor): Current input image
            new_speed (torch.Tensor): Current speed value
            feature_queue (torch.Tensor): Historical feature context (None for training mode)

        Returns:
            tuple: (acc_output, steering_output, new_feature)
                - acc_output: Predicted acceleration control [-1, 1]
                - steering_output: Predicted steering control [-1, 1]
                - new_feature: Updated feature queue for next timestep
        """

        # Training mode vs Inference mode handling
        if self.batch_size != 1:
            # Training mode: Process entire batch, no temporal context needed
            feature_queue = None
            feature_queue = self.cnnvit(new_img, new_speed)  # Shape: (batch, 2, 256)
            # Reshape for temporal processing: (64, 2, 256) -> (1, 128, 256)
            feature_queue = feature_queue.view(1, -1, 256)
        else:
            # Inference mode: Maintain sliding window of temporal features
            new_feature = self.cnnvit(new_img, new_speed)  # Shape: (1, 2, 256)

            if feature_queue is None:
                # Cold start: Initialize feature queue with zeros and add current features
                feature_queue = torch.zeros(
                    1, self.tokens_len, self.embed_dim, device=new_feature.device
                )
                # Place current features at the end of the queue
                feature_queue[:, -2:, :] = new_feature
            else:
                # Sliding window: Remove oldest features and add current features
                # feature_queue shape: (batch, 128, 256)
                feature_queue = torch.cat((feature_queue[:, 2:, :], new_feature), dim=1)

        # Store current feature state for next iteration
        new_feature = feature_queue

        # Add temporal positional embeddings to provide sequence order information
        feature_queue = feature_queue + self.timepositional_embedding

        # Process through temporal transformer stack
        for block in self.timetransformer:
            feature_queue = block(feature_queue)

        # Prepare for parallel branch processing
        # Transpose for convolution: (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        feature_queue = feature_queue.transpose(1, 2)  # Shape: (batch, 256, 128)

        # Apply residual downsampling to compress sequence
        feature_queue = self.downsample(feature_queue)  # Shape: (batch, 256, 64)

        # Acceleration prediction branch
        acc_branch = self.acc_branch1(feature_queue)  # Shape: (batch, 256, 1)
        acc_branch = acc_branch.transpose(1, 2)  # Shape: (batch, 1, 256)
        acc_branch = self.acc_branch2(acc_branch)  # Shape: (batch, 1, 128)

        # Steering prediction branch (parallel processing)
        steering_branch = self.steering_branch1(feature_queue)  # Shape: (batch, 256, 1)
        steering_branch = steering_branch.transpose(1, 2)  # Shape: (batch, 1, 256)
        steering_branch = self.steering_branch2(
            steering_branch
        )  # Shape: (batch, 1, 128)

        # Combine both branches for joint feature processing
        combined_branch = torch.cat(
            (acc_branch, steering_branch), dim=2
        )  # Shape: (batch, 1, 256)
        combined_branch = self.combined_branch(
            combined_branch
        )  # Shape: (batch, 1, 256)

        # Generate final control predictions with bounded output
        acc_output = self.acc_outputlayer(combined_branch)  # Shape: (batch, 1, 1)
        steering_output = self.steering_outputlayer(
            combined_branch
        )  # Shape: (batch, 1, 1)

        return (
            acc_output,  # Acceleration control: (batch, 1, 1)
            steering_output,  # Steering control: (batch, 1, 1)
            new_feature,  # Updated feature queue: (batch, 128, 256)
        )
