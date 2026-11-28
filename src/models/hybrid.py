"""Hybrid model: 3D-CNN features + a Transformer over the spectral bands."""

import torch
import torch.nn as nn


class HybridCNNViT(nn.Module):
    def __init__(self, n_bands=10, n_classes=2, dim=16):
        super().__init__()
        # 3D-CNN: local spatial-spectral features, one feature vector per band
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((n_bands, 1, 1)),  # (B, dim, bands, 1, 1)
        )
        # Each band becomes one token; position embedding keeps band order
        self.pos = nn.Parameter(torch.zeros(1, n_bands, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x shape: (batch, 1, bands, height, width)
        x = self.cnn(x)
        x = x.flatten(2).transpose(1, 2)  # (batch, bands, dim)
        x = x + self.pos
        x = self.transformer(x)  # self-attention across bands
        return self.classifier(x.mean(dim=1))


if __name__ == "__main__":
    model = HybridCNNViT()
    dummy = torch.randn(4, 1, 10, 9, 9)
    print("Output shape:", model(dummy).shape)