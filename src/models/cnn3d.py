"""A small 3D-CNN that classifies a pixel patch as healthy or stressed."""

import torch
import torch.nn as nn


class Small3DCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # squeeze to (16, 1, 1, 1)
        )
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x):
        # x shape: (batch, 1, bands, height, width)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    model = Small3DCNN()
    dummy = torch.randn(4, 1, 10, 9, 9)  # 4 patches, 10 bands, 9x9 pixels
    output = model(dummy)
    print("Output shape:", output.shape)