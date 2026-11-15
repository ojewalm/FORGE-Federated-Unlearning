"""ResNet-18 with a splittable head for FCL and disjoint-subspace routing."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision.models import resnet18


class SplitHeadResNet(nn.Module):
    """ResNet-18 with the backbone exposed as `features` and a linear `head`.

    The split is what enables the disjoint-subspace parallel-composition regime
    of Theorem 3: decay acts on `features`, summary subtraction on `head`.
    """

    def __init__(self, num_classes: int, in_channels: int = 3, small_input: bool = False):
        super().__init__()
        base = resnet18(weights=None, num_classes=num_classes)
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if small_input:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
        # Split: features = everything up to and including avgpool; head = fc
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten(),
        )
        self.head = base.fc  # Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def head_parameters(self):
        return self.head.parameters()

    def feature_parameters(self):
        return self.features.parameters()

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


def build_resnet18(num_classes: int, in_channels: int = 3, small_input: bool = False) -> SplitHeadResNet:
    return SplitHeadResNet(num_classes=num_classes, in_channels=in_channels, small_input=small_input)
