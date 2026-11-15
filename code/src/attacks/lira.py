"""LiRA: likelihood-ratio membership-inference attack (Carlini et al. S&P 2022).

Full LiRA trains a fleet of shadow models to estimate the per-sample
loss distribution under both ``in'' and ``out'' hypotheses, then uses a
likelihood ratio as the membership score. The reference implementation
uses 128+ shadow models. On M4-class hardware this is prohibitive, so we
implement a reduced-shadow variant: by default `n_shadow=8`, which has
been reported in follow-up work \citep{liu2022mia} to retain most of the
attack's discriminative power while being tractable.

Attack protocol:
  1. For each shadow model s in 1..n_shadow, sample a random "in-set" I_s
     from the available training pool. Train the shadow model on I_s.
  2. For each query sample x_q, compute per-sample loss ell_s(x_q) across
     all shadow models.
  3. Separate per-sample losses into those where x_q in I_s (``in''
     distribution) and x_q not in I_s (``out'' distribution).
  4. Fit a Gaussian to each (mu_in, sigma_in) and (mu_out, sigma_out).
  5. For the target model, compute ell_target(x_q) and the log-likelihood
     ratio score Lambda(x_q) = log N(ell_target; mu_out, sigma_out^2)
                             - log N(ell_target; mu_in,  sigma_in^2).
  6. Report AUROC and TPR @ low FPR as attack metrics.

We expose a `n_shadow` parameter that defaults to a tractable value for
M4; raising it monotonically improves attack strength at compute cost.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


logger = logging.getLogger("fu")


@torch.no_grad()
def _per_sample_loss(model: nn.Module, dataset: Dataset, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    device_t = torch.device(device)
    losses: list[np.ndarray] = []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = _to_device(x, device_t)
        y = torch.as_tensor(y, device=device_t, dtype=torch.long)
        logits = model(x)
        per_ex = F.cross_entropy(logits, y, reduction="none")
        losses.append(per_ex.detach().cpu().numpy())
    return np.concatenate(losses) if losses else np.array([])


def _train_shadow(
    shadow_dataset: Dataset,
    model_factory: Callable[[], nn.Module],
    device: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.02,
) -> nn.Module:
    """Train a single shadow model on `shadow_dataset`."""
    device_t = torch.device(device)
    model = model_factory().to(device_t)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(shadow_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x = _to_device(x, device_t)
            y = torch.as_tensor(y, device=device_t, dtype=torch.long)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model


def prepare_lira_shadows(
    *,
    pool_dataset: Dataset,
    query_members: Subset,
    query_nonmembers: Subset,
    model_factory: Callable[[], nn.Module],
    device: str = "cpu",
    n_shadow: int = 8,
    shadow_frac: float = 0.5,
    shadow_epochs: int = 10,
    seed: int = 0,
) -> dict:
    """Train the LiRA shadow fleet once; return per-shadow losses and in-masks.

    This is factored out so that benchmark.py can call it once and reuse the
    returned structure for every target model, instead of retraining shadows
    per method.
    """
    rng = np.random.default_rng(seed)
    pool_size = len(pool_dataset)
    in_size = int(pool_size * shadow_frac)

    all_queries = Subset(
        pool_dataset,
        list(query_members.indices) + list(query_nonmembers.indices),
    )
    query_pool_idx = np.array(list(query_members.indices) + list(query_nonmembers.indices))
    n_queries = len(query_pool_idx)

    all_losses = np.zeros((n_shadow, n_queries), dtype=np.float32)
    all_in_mask = np.zeros((n_shadow, n_queries), dtype=bool)

    for s in range(n_shadow):
        in_idx = rng.choice(pool_size, size=in_size, replace=False)
        in_set = set(int(i) for i in in_idx)
        shadow_ds = Subset(pool_dataset, in_idx.tolist())
        shadow = _train_shadow(shadow_ds, model_factory, device, epochs=shadow_epochs)
        losses = _per_sample_loss(shadow, all_queries, device)
        all_losses[s] = losses
        all_in_mask[s] = np.array([int(idx) in in_set for idx in query_pool_idx])
        logger.info(f"[lira] shadow {s+1}/{n_shadow} trained; mean loss {losses.mean():.4f}")

    return {
        "all_losses": all_losses,
        "all_in_mask": all_in_mask,
        "query_pool_idx": query_pool_idx,
        "n_members": len(query_members),
        "n_nonmembers": len(query_nonmembers),
        "pool_dataset": pool_dataset,
    }


def mia_lira_score(
    *,
    target_model: nn.Module,
    shadow_bundle: dict,
    device: str = "cpu",
) -> dict:
    """Score a target model against a pre-prepared LiRA shadow bundle.

    Returns the same metrics as `mia_lira` but reuses the shadows.
    """
    all_losses = shadow_bundle["all_losses"]
    all_in_mask = shadow_bundle["all_in_mask"]
    query_pool_idx = shadow_bundle["query_pool_idx"]
    n_members = shadow_bundle["n_members"]
    n_nonmembers = shadow_bundle["n_nonmembers"]
    pool_dataset = shadow_bundle["pool_dataset"]
    n_queries = len(query_pool_idx)

    scores = np.zeros(n_queries, dtype=np.float32)
    # Batch per-sample loss computation for the target model.
    target_ds = Subset(pool_dataset, query_pool_idx.tolist())
    target_losses = _per_sample_loss(target_model, target_ds, device)

    # Score magnitude is clipped to CLIP_SCORE to keep the sum numerically
    # finite — degenerate Gaussian fits (very small sigma_in or sigma_out)
    # otherwise blow log-likelihood ratios to inf. The ROC-based advantage
    # is monotone-invariant in the score ordering, so clipping doesn't
    # affect the attack metric.
    CLIP_SCORE = 50.0
    n_overflow = 0
    for q in range(n_queries):
        l_in = all_losses[all_in_mask[:, q], q]
        l_out = all_losses[~all_in_mask[:, q], q]
        if len(l_in) < 2 or len(l_out) < 2:
            scores[q] = 0.0
            continue
        # Floor std to avoid degenerate fits (shadow fleet all correct/wrong).
        mu_in, sd_in = float(l_in.mean()), max(float(l_in.std()), 1e-3)
        mu_out, sd_out = float(l_out.mean()), max(float(l_out.std()), 1e-3)
        l_target = float(target_losses[q])
        log_p_in = -0.5 * ((l_target - mu_in) / sd_in) ** 2 - math.log(sd_in)
        log_p_out = -0.5 * ((l_target - mu_out) / sd_out) ** 2 - math.log(sd_out)
        raw = log_p_in - log_p_out
        if abs(raw) > CLIP_SCORE:
            n_overflow += 1
        scores[q] = float(np.clip(raw, -CLIP_SCORE, CLIP_SCORE))
    if n_overflow:
        logger.info(f"[lira] clipped {n_overflow}/{n_queries} degenerate scores to +/-{CLIP_SCORE}")

    labels = np.concatenate([np.ones(n_members), np.zeros(n_nonmembers)])
    return _attack_metrics(scores, labels)


def mia_lira(
    *,
    target_model: nn.Module,
    pool_dataset: Dataset,
    query_members: Subset,
    query_nonmembers: Subset,
    model_factory: Callable[[], nn.Module],
    device: str = "cpu",
    n_shadow: int = 8,
    shadow_frac: float = 0.5,
    shadow_epochs: int = 10,
    seed: int = 0,
) -> dict:
    """Legacy entry: trains shadows and scores in one call. Use
    `prepare_lira_shadows` + `mia_lira_score` for efficiency when
    attacking multiple target models with the same shadow fleet."""
    bundle = prepare_lira_shadows(
        pool_dataset=pool_dataset,
        query_members=query_members,
        query_nonmembers=query_nonmembers,
        model_factory=model_factory,
        device=device,
        n_shadow=n_shadow,
        shadow_frac=shadow_frac,
        shadow_epochs=shadow_epochs,
        seed=seed,
    )
    return mia_lira_score(target_model=target_model, shadow_bundle=bundle, device=device)


def _attack_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute AUROC, TPR @ FPR=0.01 / 0.1, and advantage = 2*AUROC-1."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / max(labels.sum(), 1)
    fpr = fp / max((1 - labels).sum(), 1)
    # Area under ROC via trapezoid.
    auroc = float(np.trapz(tpr, fpr))
    # TPR at specific FPR thresholds.
    def tpr_at_fpr(target):
        idx = np.searchsorted(fpr, target)
        return float(tpr[min(idx, len(tpr) - 1)]) if len(tpr) else 0.0
    return {
        "auroc": auroc,
        "advantage": abs(2 * auroc - 1),
        "tpr_at_fpr_0_01": tpr_at_fpr(0.01),
        "tpr_at_fpr_0_1": tpr_at_fpr(0.1),
        "n_shadow": int(len(np.unique(np.where(np.isfinite(scores))))),
    }


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
