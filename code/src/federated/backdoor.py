"""Backdoor injection for the adversarial-departure experiment (R3).

Model: an adversarial departing client inserts a trigger-pattern backdoor into
its local training data during a configurable window of rounds immediately
preceding its erasure request. The trigger is a 4x4 bright patch in the
bottom-right corner of the image; the target label is class 0. After training
the attacker requests erasure via the standard FORGE protocol and checks whether
the backdoor survives.

Attack success rate (ASR) is measured on clean test images with the trigger
patch overlaid: the fraction classified as the target class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class BackdoorConfig:
    client_id: int                  # which client is the attacker
    rounds_before_erasure: int      # inject during the last K rounds
    target_label: int = 0
    patch_size: int = 4             # pixels; 4x4 bright patch
    patch_value: float = 2.5        # post-normalization value (bright white)


def _apply_trigger(x: torch.Tensor, cfg: BackdoorConfig) -> torch.Tensor:
    """Overlay the trigger patch in the bottom-right corner."""
    if x.dim() == 4:  # batch
        x = x.clone()
        x[..., -cfg.patch_size:, -cfg.patch_size:] = cfg.patch_value
    elif x.dim() == 3:  # single sample CHW
        x = x.clone()
        x[..., -cfg.patch_size:, -cfg.patch_size:] = cfg.patch_value
    return x


class PoisonedDataset(Dataset):
    """Wraps a client's local dataset so that every sample has the trigger overlaid
    and the label flipped to cfg.target_label. Used in the adversarial-departure
    windows only (not during the normal training rounds)."""

    def __init__(self, base: Dataset, cfg: BackdoorConfig):
        self.base = base
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        if isinstance(x, dict):
            return x, self.cfg.target_label
        x_poisoned = _apply_trigger(x, self.cfg)
        return x_poisoned, self.cfg.target_label


def should_inject(cfg: BackdoorConfig | None, client_id: int, round_index: int, total_rounds: int) -> bool:
    """Return True iff the client should see poisoned data this round."""
    if cfg is None or client_id != cfg.client_id:
        return False
    return round_index >= total_rounds - cfg.rounds_before_erasure


def attack_success_rate(model, test_loader, cfg: BackdoorConfig, device) -> float:
    """Fraction of test samples (whose true label is not the target) that are
    classified as the target class when the trigger is overlaid."""
    model.eval()
    correct_hits = 0
    n_eval = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if isinstance(x, dict):
                continue
            x = x.to(device)
            y = torch.as_tensor(y, device=device)
            mask = y != cfg.target_label
            if mask.sum() == 0:
                continue
            x_m = x[mask]
            y_m = y[mask]
            x_trig = _apply_trigger(x_m, cfg)
            preds = model(x_trig).argmax(dim=-1)
            correct_hits += (preds == cfg.target_label).sum().item()
            n_eval += int(mask.sum().item())
    return correct_hits / max(1, n_eval)
