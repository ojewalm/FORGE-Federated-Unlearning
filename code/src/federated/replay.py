"""Per-client memory buffer for replay-enabled federated continual learning.

Each client holds a FIFO buffer of (x, y) samples drawn from its past-task data.
When the orchestrator calls `effective_dataset(cid, task_ds)`, the current-task
dataset is concatenated with the client's buffer so local training sees both.
After local training, a slice of the current task's samples is admitted to the
buffer (FIFO eviction when capacity is exceeded).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


class _TupleDataset(Dataset):
    """Wraps a list of (x, y) tuples as a torch Dataset."""

    def __init__(self, items: list[tuple[Any, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class ReplayBuffer:
    """FIFO memory buffer per client, capped at `capacity_per_client` samples."""

    def __init__(self, capacity_per_client: int = 20, admit_fraction: float = 0.5, seed: int = 0):
        self.capacity = int(capacity_per_client)
        self.admit_fraction = float(admit_fraction)
        self.buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.capacity))
        self.rng = np.random.default_rng(seed)

    def admit(self, cid: int, task_ds: Dataset) -> None:
        """Sample `admit_fraction * capacity` items from task_ds into the client's buffer."""
        if self.capacity == 0 or self.admit_fraction == 0.0:
            return
        n_admit = max(1, int(round(self.capacity * self.admit_fraction)))
        n_admit = min(n_admit, len(task_ds))
        if n_admit == 0:
            return
        idx = self.rng.choice(len(task_ds), size=n_admit, replace=False)
        buf = self.buffers[cid]
        for i in idx:
            buf.append(task_ds[int(i)])

    def effective_dataset(self, cid: int, task_ds: Dataset) -> Dataset:
        """Return task_ds concatenated with the client's buffer. If the buffer is empty, returns task_ds."""
        buf = self.buffers.get(cid)
        if not buf:
            return task_ds
        buffer_ds = _TupleDataset(list(buf))
        return ConcatDataset([task_ds, buffer_ds])

    def size(self, cid: int) -> int:
        return len(self.buffers.get(cid, []))
