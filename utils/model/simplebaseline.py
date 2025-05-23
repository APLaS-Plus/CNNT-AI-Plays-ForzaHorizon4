import torch
import torch.nn as nn


class SimpleBaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 简单的CNN特征提取器，保留更多细节
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2),  # 240x144 -> 120x72
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),  # 120x72 -> 60x36
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 60x36 -> 30x18
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),  # 固定输出尺寸
        )

        # 简单的回归头
        self.regressor = nn.Sequential(
            nn.Linear(128 * 6 * 6 + 1, 256),  # +1 for speed
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # steering, acceleration
        )

    def forward(self, image, speed):
        features = self.features(image)
        features = features.view(features.size(0), -1)
        combined = torch.cat([features, speed], dim=1)
        return self.regressor(combined)
