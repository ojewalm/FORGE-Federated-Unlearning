"""Split-Tiny-ImageNet loader.

Tiny-ImageNet-200 is not shipped with torchvision. This loader expects the
standard extract at `<data_root>/tiny-imagenet-200/{train,val}/...`.
If missing, `prepare.py` will download and extract it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .base import FCLSplit
from .partition import dirichlet_partition, task_partition_by_class


TIN_MEAN = (0.4802, 0.4481, 0.3975)
TIN_STD = (0.2770, 0.2691, 0.2821)


def _transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TIN_MEAN, TIN_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])


def _val_as_imagefolder(val_root: Path) -> Path:
    """Restructure Tiny-ImageNet val/ into ImageFolder layout, once, idempotently."""
    images_dir = val_root / "images"
    if not images_dir.exists() and any((val_root / d).is_dir() for d in os.listdir(val_root)):
        return val_root  # already restructured
    annotations = val_root / "val_annotations.txt"
    if not annotations.exists():
        raise FileNotFoundError(f"Missing Tiny-ImageNet val annotations at {annotations}")
    with annotations.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, wnid = parts[0], parts[1]
            target_dir = val_root / wnid / "images"
            target_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / fname
            dst = target_dir / fname
            if src.exists() and not dst.exists():
                src.rename(dst)
    # Optionally clean up empty images/
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
    return val_root


def build_tiny_imagenet_fcl(
    data_root: str = "./data",
    num_clients: int = 100,
    num_tasks: int = 20,
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
) -> FCLSplit:
    root = Path(data_root) / "tiny-imagenet-200"
    if not root.exists():
        raise FileNotFoundError(
            f"Tiny-ImageNet not found at {root}. Run `python -m src.datasets.prepare "
            f"--dataset tiny_imagenet` first."
        )
    train_root = root / "train"
    val_root = _val_as_imagefolder(root / "val")

    train = ImageFolder(str(train_root), transform=_transforms(True))
    test = ImageFolder(str(val_root), transform=_transforms(False))

    labels = np.array(train.targets)
    client_indices = dirichlet_partition(labels, num_clients, dirichlet_alpha, seed=seed)
    task_classes = task_partition_by_class(labels, num_tasks, seed=seed)

    return FCLSplit(
        train_set=train,
        test_set=test,
        client_indices=client_indices,
        task_classes=task_classes,
        modality="image",
        num_classes=200,
        input_shape=(3, 64, 64),
        metadata={"name": "tiny_imagenet", "dirichlet_alpha": dirichlet_alpha},
    )
