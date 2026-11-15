"""Split-CIFAR-10 loader with Dirichlet non-IID partitioning."""
from __future__ import annotations

import numpy as np
from torchvision import datasets, transforms

from .base import FCLSplit
from .partition import dirichlet_partition, task_partition_by_class


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_cifar10_fcl(
    data_root: str = "./data",
    num_clients: int = 100,
    num_tasks: int = 5,
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
) -> FCLSplit:
    train = datasets.CIFAR10(data_root, train=True, download=True, transform=_transforms(True))
    test = datasets.CIFAR10(data_root, train=False, download=True, transform=_transforms(False))

    labels = np.array(train.targets)
    client_indices = dirichlet_partition(labels, num_clients, dirichlet_alpha, seed=seed)
    task_classes = task_partition_by_class(labels, num_tasks, seed=seed)

    return FCLSplit(
        train_set=train,
        test_set=test,
        client_indices=client_indices,
        task_classes=task_classes,
        modality="image",
        num_classes=10,
        input_shape=(3, 32, 32),
        metadata={"name": "cifar10", "dirichlet_alpha": dirichlet_alpha},
    )
