"""Tiny 2-conv CNN for CPU smoke tests. Not for benchmark reporting."""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """~50k params. Adapts to 28x28 or 32x32 inputs; grayscale or RGB."""

    def __init__(self, num_classes: int, in_channels: int = 3, input_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_feat = self.features(torch.zeros(1, in_channels, input_size, input_size)).shape[1]
        self.head = nn.Linear(n_feat, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def head_parameters(self):
        return self.head.parameters()

    def feature_parameters(self):
        return self.features.parameters()

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


def build_tiny_cnn(num_classes: int, in_channels: int = 3, input_size: int = 32) -> TinyCNN:
    return TinyCNN(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
