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

    def __init__(self, seq_len, embed_dim):
        super(SingleEncoder, self).__init__()
        self.seq_len = seq_len
        self.fill_seq = nn.Linear(1, seq_len)
        self.embed_dim = embed_dim
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, self.embed_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.fill_seq(x)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        return x


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, channel)
        w = self.fc(x.mean(dim=1))  # (batch, channel)
        return x * w.unsqueeze(1)


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
            CBAM(32),  # 120*80*32
            Conv2dBlock(32, 64),  # 60*40*64
            Bottleneck(64, 64),  # 60*40*64
            CBAM(64),  # 60*40*64
            Conv2dBlock(64, 128),  # 30*20*128
            Bottleneck(128, 128),  # 30*20*128
            CBAM(128),  # 30*20*128
            Conv2dBlock(128, 256),  # 15*10*256
            Bottleneck(256, 256),  # 15*10*256
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
        feature = x.view(batch_size, channels, -1).permute(0, 2, 1)
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
        self.embedding_dim1 = embedding_dim

        if self.embedding_dim1 % num_heads != 0:
            raise ValueError(
                f"embedding_dim1 ({self.embedding_dim1}) must be divisible by num_heads ({num_heads})"
            )

        # Feature extractor
        self.feature_extractor = Feature_extractor(
            input_w, input_h, self.embedding_dim1
        )

        # Image encoder
        patch_w = 3
        patch_h = 2
        self.image_encoder = VitImageEncoder(patch_w, patch_h, self.embedding_dim1)

        # Single encoder for speed, 1 -> 2, embedding_dim1
        self.speed_encoder = SingleEncoder(2, self.embedding_dim1)

        # Transformer blocks1
        self.transformer_blocks1 = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embedding_dim1, num_heads),
                    GLU(self.embedding_dim1, self.embedding_dim1),
                )
                for _ in range(num_of_simgle_layers)
            ]
        )

        # Downsample1: 8, embedding_dim1 -> 4, embedding_dim1//2
        self.downsample1_1 = nn.Conv1d(self.embedding_dim1, self.embedding_dim1, 2, 2)
        self.downsample1_2 = nn.Sequential(
            nn.LayerNorm(self.embedding_dim1),
            nn.GELU(),
            nn.Linear(self.embedding_dim1, self.embedding_dim1 // 2),
            nn.GELU(),
        )

        # Transformer blocks2
        self.embedding_dim2 = self.embedding_dim1 // 2

        if self.embedding_dim2 % num_heads != 0:
            raise ValueError(
                f"embedding_dim2 ({self.embedding_dim2}) must be divisible by num_heads ({num_heads})"
            )

        self.transformer_blocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embedding_dim2, num_heads),
                    GLU(self.embedding_dim2, self.embedding_dim2),
                )
                for _ in range(num_of_simgle_layers)
            ]
        )

        # Downsample2: 4, embedding_emb2 -> 2, embedding_emb2//2
        self.downsample2_1 = nn.Conv1d(self.embedding_dim2, self.embedding_dim2, 2, 2)
        self.downsample2_2 = nn.Sequential(
            nn.LayerNorm(self.embedding_dim2),
            nn.GELU(),
            nn.Linear(self.embedding_dim2, self.embedding_dim2 // 2),
            nn.GELU(),
        )

    def forward(self, img, speed):
        img_features = self.image_encoder(self.feature_extractor(img))
        speed_tokens = self.speed_encoder(speed)

        combined_token = torch.cat([img_features, speed_tokens], dim=1)

        output = combined_token
        for block in self.transformer_blocks1:
            output = block(output)

        output = output.transpose(1, 2)
        output = self.downsample1_1(output)
        output = output.transpose(1, 2)
        output = self.downsample1_2(output)

        for block in self.transformer_blocks2:
            output = block(output)

        output = output.transpose(1, 2)
        output = self.downsample2_1(output)
        output = output.transpose(1, 2)
        output = self.downsample2_2(output)

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
    def __init__(
        self,
        input_len=2,
        seq_len=64,
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        batch_size=64,
    ):
        super(CNN_Transformer, self).__init__()
        self.input_len = input_len
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokens_len = self.seq_len * self.input_len  # 64*2 = 128
        self.batch_size = batch_size

        self.cnnvit = CNNViTBlock()  # output: batch, 2, 256

        # 128 tokens, 256 dim
        self.timepositional_embedding = nn.Parameter(
            torch.randn(1, self.tokens_len, self.embed_dim)
        )

        self.timetransformer1 = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embed_dim, self.num_heads),
                    GLU(self.embed_dim, self.embed_dim),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.downsample1 = ResDownsample(self.embed_dim)

        # 64 tokens, 256 dim
        self.timetransformer2 = nn.ModuleList(
            [
                nn.Sequential(
                    TransformerBlock(self.embed_dim, self.num_heads),
                    GLU(self.embed_dim, self.embed_dim),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.acc_branch1 = nn.Sequential(
            nn.Linear(self.tokens_len // 2, self.tokens_len),
            nn.GELU(),
            nn.Linear(self.tokens_len, 1),
        )
        self.acc_branch2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim // 2),
        )

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

        self.combined_branch = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
        )

        self.acc_outputlayer = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Tanh(),
        )

        self.steering_outputlayer = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, new_img, new_speed, feature_queue):

        # When batch_size = 1, it means that we are training the model. In a batch, the img is following the order

        if self.batch_size != 1:
            feature_queue = None
            feature_queue = self.cnnvit(new_img, new_speed)  # batch, 2, 256
            feature_queue = feature_queue.view(1, -1, 256)  # 64, 2, 256 -> 1, 128, 256
        else:
            # predict mode, feature_queue is not None
            # img_queue: 1, 128, 256
            new_feature = self.cnnvit(new_img, new_speed)  # 1, 2, 256

            if feature_queue is None:
                # warm up
                feature_queue = torch.zeros(
                    1, self.tokens_len, self.embed_dim, device=new_feature.device
                )
                feature_queue[:, -2:, :] = new_feature
            else:
                # feature_queue: batch, 128, 256
                feature_queue = torch.cat((feature_queue[:, 2:, :], new_feature), dim=1)

        new_feature = feature_queue

        feature_queue = feature_queue + self.timepositional_embedding
        for block in self.timetransformer1:
            feature_queue = block(feature_queue)

        feature_queue = feature_queue.transpose(1, 2)  # batch, 256, 128
        feature_queue = self.downsample1(feature_queue)
        feature_queue = feature_queue.transpose(1, 2)

        # feature_queue: batch, 64, 256
        for block in self.timetransformer2:
            feature_queue = block(feature_queue)

        # feature_queue: batch, 256, 64
        feature_queue = feature_queue.transpose(1, 2)

        # Speed branch
        acc_branch = self.acc_branch1(feature_queue)  # batch, 256, 1
        acc_branch = acc_branch.transpose(1, 2)  # batch, 1, 256
        acc_branch = self.acc_branch2(acc_branch)  # batch, 1, 128

        # Steering branch
        steering_branch = self.steering_branch1(feature_queue)
        steering_branch = steering_branch.transpose(1, 2)
        steering_branch = self.steering_branch2(steering_branch)

        # Combine branches
        combined_branch = torch.cat((acc_branch, steering_branch), dim=2)
        combined_branch = self.combined_branch(combined_branch)

        # Speed output
        acc_output = self.acc_outputlayer(combined_branch)

        # Steering output
        steering_output = self.steering_outputlayer(combined_branch)

        return (
            acc_output,
            steering_output,
            new_feature,
        )  # batch, 1, 1; batch, 1, 1; batch, 128, 256
