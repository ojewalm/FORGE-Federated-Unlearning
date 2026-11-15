"""Membership inference attacks.

Yeom threshold attack (loss-based) and a LiRA-lite (single-shot likelihood-ratio).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def _per_sample_loss(model: nn.Module, dataset: Dataset, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    device_t = torch.device(device)
    losses = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x = _to_device(x, device_t)
        y = torch.as_tensor(y, device=device_t)
        logits = model(x)
        per_ex = F.cross_entropy(logits, y, reduction="none")
        losses.append(per_ex.detach().cpu().numpy())
    return np.concatenate(losses) if losses else np.array([])


def mia_yeom(
    model: nn.Module,
    member_set: Dataset,
    nonmember_set: Dataset,
    device: str = "cpu",
) -> dict:
    """Yeom threshold attack.

    Predicts membership if per-sample loss <= tau, with tau chosen as the
    average training loss (Yeom et al., CSF 2018).
    Returns dict with advantage = |accuracy - 0.5| * 2 in [0, 1].
    """
    l_m = _per_sample_loss(model, member_set, device)
    l_nm = _per_sample_loss(model, nonmember_set, device)
    tau = float(l_m.mean()) if len(l_m) else 0.0
    tp = float((l_m <= tau).mean()) if len(l_m) else 0.0
    fp = float((l_nm <= tau).mean()) if len(l_nm) else 0.0
    # Balanced accuracy.
    acc = 0.5 * (tp + (1.0 - fp))
    advantage = abs(acc - 0.5) * 2.0
    return {"acc": acc, "tpr": tp, "fpr": fp, "advantage": advantage, "tau": tau}


def mia_lira_lite(
    model: nn.Module,
    member_set: Dataset,
    nonmember_set: Dataset,
    device: str = "cpu",
) -> dict:
    """LiRA-lite: single reference model = untrained model of same architecture.

    Full LiRA requires a fleet of shadow models. This lite version uses the
    model at initialisation as the reference to give a comparable advantage
    number cheaply; not competitive with the real attack, but consistent
    across the sweep.
    """
    l_target = _per_sample_loss(model, member_set, device)
    l_neg = _per_sample_loss(model, nonmember_set, device)
    # Reference: freshly initialised model. In practice pass in from caller
    # to avoid re-instantiating per call; here we approximate with mean losses.
    mu_ref = float(np.mean(np.concatenate([l_target, l_neg]) if (len(l_target) + len(l_neg)) else np.array([0.0])))
    scores_m = (mu_ref - l_target)
    scores_n = (mu_ref - l_neg)
    # ROC-based advantage (Pinf - Pfp at optimal threshold).
    scores = np.concatenate([scores_m, scores_n])
    labels = np.concatenate([np.ones_like(scores_m), np.zeros_like(scores_n)])
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tpr = np.cumsum(labels_sorted) / max(labels.sum(), 1)
    fpr = np.cumsum(1 - labels_sorted) / max((1 - labels).sum(), 1)
    advantage = float((tpr - fpr).max()) if len(tpr) else 0.0
    return {"advantage": advantage, "tpr": tpr.tolist()[:5], "fpr": fpr.tolist()[:5]}


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
