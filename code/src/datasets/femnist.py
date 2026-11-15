"""FEMNIST loader.

Uses the LEAF-derived FEMNIST partitioning (writer-per-client). This loader
prefers the `femnist` dataset from HuggingFace (`flwrlabs/femnist`) which
provides pre-processed 28x28 grayscale images and writer_id. Falls back to
raw LEAF JSON if present under `<data_root>/femnist/`.

Continual tasks: 6 groups of characters (uppercase A-I, J-R, S-Z; lowercase
a-i, j-r, s-z; digits merged into groups). Standard FEMNIST label mapping:
    0-9 : digits
    10-35: uppercase A-Z
    36-61: lowercase a-z
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from .base import FCLSplit, LabeledArrayDataset
from .partition import natural_partition


# Task groupings by label index into (uppercase, lowercase, digits) buckets.
FEMNIST_TASK_GROUPS: list[list[int]] = [
    list(range(10, 19)),           # A-I
    list(range(19, 28)),           # J-R
    list(range(28, 36)),           # S-Z
    list(range(36, 45)),           # a-i
    list(range(45, 54)),           # j-r
    list(range(54, 62)) + list(range(0, 10)),  # s-z + digits
]


class _FEMNISTArray(LabeledArrayDataset):
    """Wraps FEMNIST arrays into a torch dataset with train-time augmentations."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.targets = self.labels.tolist()
        self.train = train
        self.tform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = (self.images[idx] * 255).astype(np.uint8)
        if img.ndim == 2:
            img = img[..., None]  # HxWx1
        return self.tform(img), int(self.labels[idx])


def _load_femnist_hf(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FEMNIST from HuggingFace 'flwrlabs/femnist'. Returns train/test arrays + writer ids."""
    from datasets import load_dataset  # local import so package is optional

    cache = data_root / "hf_cache"
    ds = load_dataset("flwrlabs/femnist", cache_dir=str(cache))
    # Combine into arrays. HF split names may vary; adapt if needed.
    train = ds["train"]
    test = ds.get("test", None)
    if test is None:
        # 90/10 by writer
        writers = np.array(train["writer_id"])
        uniq_writers = np.unique(writers)
        rng = np.random.default_rng(0)
        rng.shuffle(uniq_writers)
        cut = int(0.9 * len(uniq_writers))
        train_writers = set(uniq_writers[:cut].tolist())
        train_mask = np.array([w in train_writers for w in writers])
        train_imgs = np.array([np.array(im) for im in train["image"]])
        train_labels = np.array(train["character"])
        train_wr = writers
        test_imgs = train_imgs[~train_mask]
        test_labels = train_labels[~train_mask]
        train_imgs = train_imgs[train_mask]
        train_labels = train_labels[train_mask]
        train_wr = train_wr[train_mask]
    else:
        train_imgs = np.array([np.array(im) for im in train["image"]])
        train_labels = np.array(train["character"])
        train_wr = np.array(train["writer_id"])
        test_imgs = np.array([np.array(im) for im in test["image"]])
        test_labels = np.array(test["character"])
    # Normalise labels to int
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)
    return train_imgs, train_labels, test_imgs, test_labels, train_wr


def build_femnist_fcl(
    data_root: str = "./data",
    num_clients: int | None = None,
    seed: int = 0,
    subsample_writers: int | None = None,
) -> FCLSplit:
    """Build FEMNIST FCL split.

    Args:
        num_clients: if None, use all writers as clients. Otherwise, subsample
            this many writers deterministically.
        subsample_writers: alias for num_clients (kept for readability).
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train_imgs, train_labels, test_imgs, test_labels, writers = _load_femnist_hf(root)

    # Optionally subsample writers.
    target_writers = num_clients if num_clients is not None else subsample_writers
    if target_writers is not None:
        uniq = np.unique(writers)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        keep = set(uniq[:target_writers].tolist())
        mask = np.array([w in keep for w in writers])
        train_imgs, train_labels, writers = train_imgs[mask], train_labels[mask], writers[mask]

    train = _FEMNISTArray(train_imgs, train_labels, train=True)
    test = _FEMNISTArray(test_imgs, test_labels, train=False)

    natural = natural_partition(writers)
    # Deterministic client id ordering.
    client_indices = [natural[w] for w in sorted(natural.keys())]

    return FCLSplit(
        train_set=train,
        test_set=test,
        client_indices=client_indices,
        task_classes=FEMNIST_TASK_GROUPS,
        modality="image",
        num_classes=62,
        input_shape=(1, 28, 28),
        metadata={"name": "femnist", "num_writers": len(client_indices)},
    )
