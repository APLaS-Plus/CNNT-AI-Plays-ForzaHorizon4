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
        self.embedding = nn.Linear(1, output_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        return x


class SimpleCNNBaseline(nn.Module):
    def __init__(self, input_w=960, input_h=640):
        super(SimpleCNNBaseline, self).__init__()
        # input size: 960*640*3
        self.input_w = input_w
        self.input_h = input_h
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
            Conv(256, 512, 5, 5, 0), # 3*2*512
            Bottleneck(512, 512),  # 3*2*512
            CBAM(512),  # 3*2*512
            nn.Dropout(0.1),
        )

        self.single_encoder = SingleEncoder(512)

        conv_output_w = input_w // (2**6) //5  # 3
        conv_output_h = input_h // (2**6) //5  # 2
        conv_feature_size = conv_output_w * conv_output_h * 512  # 3*2*512 = 3072

        self.fusion_layer = nn.Sequential(
            nn.Linear(conv_feature_size + 512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),  # Output layer for regression
            nn.Tanh(),
        )

    def forward(self, x, speed):
        # x = x.view(-1, 1, self.input_h, self.input_w)
        x = self.conv(x)
        batch_size = x.size(0)
        conv_features = x.view(batch_size, -1)
        # print(f"Conv features shape: {conv_features.shape}")

        speed_encoded = self.single_encoder(speed)
        speed_encoded = speed_encoded.view(batch_size, -1)
        # print(f"Speed encoded shape: {speed_encoded.shape}")

        fused_features = torch.cat([conv_features, speed_encoded], dim=1)
        fused_features = self.fusion_layer(fused_features)

        output = self.output_layer(fused_features)
        return output
