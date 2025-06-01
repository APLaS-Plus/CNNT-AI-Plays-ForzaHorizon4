import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Conv, CBAM, SimAM, Bottleneck, GhostConv


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.op(x)
        return x


class SingleEncoder(nn.Module):
    def __init__(self, output_size):
        super(SingleEncoder, self).__init__()
        self.output_size = output_size
        self.embedding = nn.Sequential(
            nn.Linear(1, self.output_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_size // 2, self.output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        return x


class Feature_extractor(nn.Module):
    def __init__(self, input_w=960, input_h=640, embedding_dim=1024):
        super(Feature_extractor, self).__init__()
        # input size: 960*640*3
        self.input_w = input_w
        self.input_h = input_h
        self.embedding_dim = embedding_dim
        
        # 计算预期的最终尺寸
        self.expected_final_h = 2
        self.expected_final_w = 3
        
        self.conv = nn.Sequential(
            GhostConv(3, 16),
            nn.MaxPool2d(2),  # 480*320*16
            nn.Dropout(0.1),
            SimAM(),  # 480*320*16
            Conv2dBlock(16, 32, 5, 2, 2),  # 120*80*32
            Bottleneck(32, 32),  # 120*80*32
            nn.Dropout(0.1),
            CBAM(32),  # 120*80*32
            Conv2dBlock(32, 64),  # 60*40*64
            Bottleneck(64, 64),  # 60*40*64
            nn.Dropout(0.1),
            CBAM(64),  # 60*40*64
            Conv2dBlock(64, 128),  # 30*20*128
            Bottleneck(128, 128),  # 30*20*128
            nn.Dropout(0.1),
            CBAM(128),  # 30*20*128
            Conv2dBlock(128, 256),  # 15*10*256
            Bottleneck(256, 256),  # 15*10*256
            nn.Dropout(0.1),
            CBAM(256),  # 15*10*256
            Conv(256, 512, 5, 5, 0),  # 3*2*512
            Bottleneck(512, 512),  # 3*2*512
            CBAM(512),  # 3*2*512
            nn.Dropout(0.1),
        )
        self.embedding = nn.Sequential(
            nn.Linear(512, self.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # 验证输入尺寸
        if x.size(-2) != self.input_h or x.size(-1) != self.input_w:
            raise ValueError(
                f"Expected input size ({self.input_h}, {self.input_w}), got ({x.size(-2)}, {x.size(-1)})"
            )

        x = self.conv(x)
        batch_size, channels, height, width = x.size()

        # 验证输出尺寸是否符合预期
        if height != self.expected_final_h or width != self.expected_final_w:
            print(
                f"Warning: Expected final size ({self.expected_final_h}, {self.expected_final_w}), got ({height}, {width})"
            )

        # 重塑为序列格式: (batch_size, seq_len, channels)
        feature = x.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()
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

        # 修复：移除函数调用的括号
        return x + self.learned_positional_embedding


class VitBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(VitBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=0.1
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
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


class CNNViT(nn.Module):
    def __init__(
        self, input_w=960, input_h=640, embedding_dim=1024, num_layers=6, num_heads=8
    ):
        super(CNNViT, self).__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.embedding_dim = embedding_dim

        # Feature extractor
        self.feature_extractor = Feature_extractor(input_w, input_h, embedding_dim)

        # Image encoder
        patch_w = 3  # 修正计算方式
        patch_h = 2
        self.image_encoder = VitImageEncoder(patch_w, patch_h, embedding_dim)

        # Single encoder for speed
        self.single_encoder = SingleEncoder(embedding_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [VitBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        )

        # cls token, for acc and steering
        self.cls_token = nn.Parameter(torch.randn(1, 2, embedding_dim))
        # 任务特定的特征提取
        self.acc_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.steering_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 任务间通信层
        self.task_communication = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 最终输出头
        self.acc_head = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1),
            nn.Tanh(),
        )

        self.steering_head = nn.Sequential(
            nn.Linear(embedding_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, img, speed):
        # Feature extraction
        img_features = self.feature_extractor(img)

        # Image encoding
        img_features = self.image_encoder(img_features)

        # Single encoder for speed
        speed_features = self.single_encoder(speed)

        # Concatenate image and speed features
        features = torch.cat(
            (img_features, speed_features.unsqueeze(1)), dim=1
        ).contiguous()

        # Add cls token
        cls_token = self.cls_token.expand(features.size(0), -1, -1)
        features = torch.cat((cls_token, features), dim=1).contiguous()

        # Transformer blocks
        for block in self.transformer_blocks:
            features = block(features)

        # Classification head
        acc_features = features[:, 0, :]  # cls token
        steering_features = features[:, 1, :]
        acc_features = self.acc_branch(acc_features)
        steering_features = self.steering_branch(steering_features)
        combined_features = torch.cat(
            (acc_features, steering_features), dim=1
        ).contiguous()
        shared_features = self.task_communication(combined_features)
        acc_output = self.acc_head(shared_features)
        steering_output = self.steering_head(shared_features)
        output = torch.cat((acc_output, steering_output), dim=1).contiguous()
        return output
