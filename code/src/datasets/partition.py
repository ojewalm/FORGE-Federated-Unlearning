"""Client partitioning for FL: Dirichlet (synthetic non-IID) and natural (writer-based)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_per_client: int = 10,
) -> list[np.ndarray]:
    """Partition indices across clients via a Dirichlet(alpha) distribution over labels.

    Small alpha (e.g. 0.1) yields high heterogeneity; large alpha (e.g. 100) approaches IID.
    Ensures each client has at least `min_per_client` samples.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_classes = int(labels.max() + 1)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # Guard against empty/underfilled clients by transferring from largest to smallest.
    all_idx = list(range(len(labels)))
    counts = [len(ci) for ci in client_indices]
    while min(counts) < min_per_client:
        smallest = counts.index(min(counts))
        largest = counts.index(max(counts))
        transfer = client_indices[largest].pop()
        client_indices[smallest].append(transfer)
        counts = [len(ci) for ci in client_indices]

    return [np.array(ci, dtype=np.int64) for ci in client_indices]


def natural_partition(group_ids: np.ndarray) -> dict:
    """Partition indices by an existing group id column (e.g. FEMNIST writers).

    Writers may be string ids (e.g. 'f0046_20') or int ids. Group keys preserve
    the original type so callers can look up by the raw id.
    """
    out: dict = defaultdict(list)
    for i, g in enumerate(group_ids):
        out[g].append(i)
    return {g: np.array(ix, dtype=np.int64) for g, ix in out.items()}


def task_partition_by_class(
    labels: np.ndarray,
    num_tasks: int,
    seed: int = 0,
) -> list[list[int]]:
    """Split classes into `num_tasks` sequential groups. Returns list of class-lists per task."""
    n_classes = int(labels.max() + 1)
    rng = np.random.default_rng(seed)
    class_order = np.arange(n_classes)
    rng.shuffle(class_order)
    return [list(map(int, chunk)) for chunk in np.array_split(class_order, num_tasks)]
