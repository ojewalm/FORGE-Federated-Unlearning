"""Base data structures for federated continual learning."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class FCLSplit:
    """Federated continual learning dataset split.

    Attributes:
        train_set: full training dataset (indexed).
        test_set: full test dataset.
        client_indices: list of per-client index arrays into train_set.
        task_classes: list of class-lists per task.
        modality: "image" | "text".
        num_classes: total number of classes across all tasks.
        input_shape: shape of a single sample (excluding batch dim).
        metadata: extra dataset info.
    """
    train_set: Dataset
    test_set: Dataset
    client_indices: list[np.ndarray]
    task_classes: list[list[int]]
    modality: str
    num_classes: int
    input_shape: tuple[int, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def num_clients(self) -> int:
        return len(self.client_indices)

    @property
    def num_tasks(self) -> int:
        return len(self.task_classes)

    def client_dataset(self, client_id: int, task_id: int | None = None) -> Subset:
        """Return the client's training data, optionally restricted to a task's classes."""
        idx = self.client_indices[client_id]
        if task_id is None:
            return Subset(self.train_set, idx.tolist())
        classes = set(self.task_classes[task_id])
        labels = _labels_for(self.train_set, idx)
        keep = idx[np.isin(labels, list(classes))]
        return Subset(self.train_set, keep.tolist())

    def task_test_set(self, task_id: int) -> Subset:
        """Test data restricted to a task's classes."""
        classes = set(self.task_classes[task_id])
        labels = _labels_for(self.test_set, np.arange(len(self.test_set)))
        keep = np.where(np.isin(labels, list(classes)))[0]
        return Subset(self.test_set, keep.tolist())


def _labels_for(dataset: Dataset, indices: np.ndarray) -> np.ndarray:
    """Extract labels for a set of indices; supports common torchvision-shaped datasets."""
    if hasattr(dataset, "targets"):
        return np.array([int(dataset.targets[i]) for i in indices])
    labels = []
    for i in indices:
        item = dataset[int(i)]
        y = item[1] if isinstance(item, tuple) else item["label"]
        labels.append(int(y))
    return np.array(labels)


class LabeledArrayDataset(Dataset):
    """Simple in-memory dataset from parallel arrays. Handy for text/tabular."""

    def __init__(self, xs, ys, transform=None):
        self.xs = xs
        self.ys = np.asarray(ys, dtype=np.int64)
        self.targets = self.ys.tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        x = self.xs[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.ys[idx])
