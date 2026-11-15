"""Utility metrics: test accuracy, per-task accuracy, BWT forgetting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataset: Dataset, device: str = "cpu", batch_size: int = 128) -> float:
    model.eval()
    device_t = torch.device(device)
    correct = 0
    total = 0
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = _to_device(x, device_t)
        y = torch.as_tensor(y, device=device_t)
        preds = model(x).argmax(-1)
        correct += int((preds == y).sum())
        total += y.numel()
    return correct / max(total, 1)


def per_task_accuracy(model: nn.Module, fcl_split, device: str = "cpu") -> list[float]:
    return [evaluate_accuracy(model, fcl_split.task_test_set(t), device) for t in range(fcl_split.num_tasks)]


def average_test_accuracy(model: nn.Module, fcl_split, device: str = "cpu") -> float:
    accs = per_task_accuracy(model, fcl_split, device)
    return float(np.mean(accs)) if accs else 0.0


def bwt_forgetting(acc_matrix: np.ndarray) -> float:
    """Backwards transfer (Lopez-Paz & Ranzato, 2017).

    acc_matrix[i, j] = accuracy on task j after training on task i.
    BWT = mean_{j < T-1} (acc[T-1, j] - acc[j, j])
    """
    T = acc_matrix.shape[0]
    if T < 2:
        return 0.0
    vals = [acc_matrix[T - 1, j] - acc_matrix[j, j] for j in range(T - 1)]
    return float(np.mean(vals))


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
