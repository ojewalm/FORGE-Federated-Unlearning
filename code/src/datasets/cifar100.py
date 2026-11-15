"""Split-CIFAR-100 loader with Dirichlet non-IID partitioning."""
from __future__ import annotations

import numpy as np
import torch
from torchvision import datasets, transforms

from .base import FCLSplit
from .partition import dirichlet_partition, task_partition_by_class


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_cifar100_fcl(
    data_root: str = "./data",
    num_clients: int = 100,
    num_tasks: int = 10,
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
) -> FCLSplit:
    train = datasets.CIFAR100(data_root, train=True, download=True, transform=_transforms(True))
    test = datasets.CIFAR100(data_root, train=False, download=True, transform=_transforms(False))

    labels = np.array(train.targets)
    client_indices = dirichlet_partition(labels, num_clients, dirichlet_alpha, seed=seed)
    task_classes = task_partition_by_class(labels, num_tasks, seed=seed)

    return FCLSplit(
        train_set=train,
        test_set=test,
        client_indices=client_indices,
        task_classes=task_classes,
        modality="image",
        num_classes=100,
        input_shape=(3, 32, 32),
        metadata={"name": "cifar100", "dirichlet_alpha": dirichlet_alpha},
    )
