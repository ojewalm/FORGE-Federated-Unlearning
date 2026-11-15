"""Forgetting-side metrics: retrain-KL, backdoor survival, temporal forgetting."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def retrain_kl(
    model_unlearned: nn.Module,
    model_retrained: nn.Module,
    probe_set: Dataset,
    device: str = "cpu",
    batch_size: int = 128,
) -> float:
    """KL(unlearned || retrained) averaged over a probe set."""
    model_unlearned.eval(); model_retrained.eval()
    device_t = torch.device(device)
    total = 0.0
    n = 0
    for x, _ in DataLoader(probe_set, batch_size=batch_size, shuffle=False):
        x = _to_device(x, device_t)
        log_p = F.log_softmax(model_unlearned(x), dim=-1)
        log_q = F.log_softmax(model_retrained(x), dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1).sum().item()
        total += kl
        n += (x["input_ids"].shape[0] if isinstance(x, dict) else x.shape[0])
    return total / max(n, 1)


@torch.no_grad()
def backdoor_survival(
    model: nn.Module,
    trigger_set: Dataset,
    target_class: int,
    device: str = "cpu",
) -> float:
    """Fraction of trigger inputs still classified as the backdoor target class."""
    model.eval()
    device_t = torch.device(device)
    hits = 0
    total = 0
    for x, _ in DataLoader(trigger_set, batch_size=64, shuffle=False):
        x = _to_device(x, device_t)
        preds = model(x).argmax(-1)
        hits += int((preds == target_class).sum())
        total += preds.numel()
    return hits / max(total, 1)


def temporal_forgetting_metric(acc_over_rounds: np.ndarray, r_forget: int, window: int = 20) -> float:
    """TF_{i*}(k): change in accuracy on client i*'s task after erasure round.

    Measures the drop from the pre-erasure window to the post-erasure window.
    Positive means loss of accuracy after erasure (i.e. actual forgetting).
    """
    if r_forget - window < 0 or r_forget + window > len(acc_over_rounds):
        return 0.0
    pre = acc_over_rounds[r_forget - window: r_forget].mean()
    post = acc_over_rounds[r_forget: r_forget + window].mean()
    return float(pre - post)


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
