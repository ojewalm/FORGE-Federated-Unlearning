"""Small CNN for FEMNIST (LEAF-compatible)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """2 conv + 2 FC layers, matches the LEAF FEMNIST baseline architecture."""

    def __init__(self, num_classes: int = 62, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        )
        self.head = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def head_parameters(self):
        return self.head.parameters()

    def feature_parameters(self):
        return self.features.parameters()

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


def build_small_cnn(num_classes: int, in_channels: int = 1) -> SmallCNN:
    return SmallCNN(num_classes=num_classes, in_channels=in_channels)
