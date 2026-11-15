"""Pre-committed summary channel (Section 4.2 of the paper).

Four candidate summaries: class-conditional moments, distilled logits on a
public probe, DP synthetic microbatch (placeholder for a proper private GAN),
and sketched gradients.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..utils.dp import gaussian_mechanism_sigma


def _iter_batches(dataset: Dataset, batch_size: int = 64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def summarise_moments(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str,
    eps: float,
    delta: float,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Class-conditional feature moments (mean + diagonal covariance) with DP noise.

    Feature extraction uses `model.features` (SplitHeadResNet / SmallCNN /
    TextTransformer.features).
    """
    model.eval()
    device_t = torch.device(device)
    per_class_feats: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for x, y in _iter_batches(dataset):
            x = _to_device(x, device_t)
            y = torch.as_tensor(y, device=device_t)
            if hasattr(model, "features") and callable(model.features):
                feats = model.features(x) if not isinstance(x, torch.Tensor) else model.features(x)
            else:
                # Fallback: forward and take penultimate (won't be exact for all archs)
                feats = model(x)
            for c in torch.unique(y).tolist():
                mask = (y == c)
                per_class_feats.setdefault(int(c), []).append(feats[mask].cpu())

    means: dict[int, np.ndarray] = {}
    diag_covs: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    # Sensitivity of the mean under 1-sample change with clipped features (assume ||feat|| <= 1
    # after normalisation; users should normalise features before calling).
    sensitivity = 1.0
    sigma_dp = gaussian_mechanism_sigma(sensitivity, eps, delta)

    for c, chunks in per_class_feats.items():
        feats = torch.cat(chunks, dim=0)
        n = feats.shape[0]
        if n == 0:
            continue
        # Clip per-sample L2 norm to 1 for a clean sensitivity.
        norms = feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        feats_clipped = feats / norms.clamp(min=1.0)
        mu = feats_clipped.mean(dim=0)
        cov = (feats_clipped - mu).pow(2).mean(dim=0)
        # DP noise on mu and diag(cov).
        mu_dp = mu + torch.randn_like(mu) * (sigma_dp / max(n, 1))
        cov_dp = cov + torch.randn_like(cov) * (sigma_dp / max(n, 1))
        means[c] = mu_dp.numpy()
        diag_covs[c] = cov_dp.numpy()
        counts[c] = n

    return {
        "type": "moments",
        "means": means,
        "diag_covs": diag_covs,
        "counts": counts,
        "eps": eps,
        "delta": delta,
    }


def summarise_logits(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str,
    eps: float,
    delta: float,
    probe_set: Dataset | None = None,
    probe_size: int = 256,
    local_ft_steps: int = 20,
    lr: float = 0.005,
) -> dict[str, Any]:
    """Distilled logits on a public probe, after brief local fine-tuning.

    If no probe_set is given, uses the first `probe_size` samples of the
    provided dataset (a degenerate but useful default for prototyping).
    """
    model.train()
    device_t = torch.device(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Brief local fine-tuning to imprint client-specific memorisation into logits.
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    steps = 0
    for x, y in loader:
        if steps >= local_ft_steps:
            break
        x = _to_device(x, device_t)
        y = torch.as_tensor(y, device=device_t)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        steps += 1

    # Compute logits on the probe set.
    probe = probe_set if probe_set is not None else dataset
    model.eval()
    logits_chunks = []
    seen = 0
    with torch.no_grad():
        for x, _ in _iter_batches(probe, batch_size=64):
            if seen >= probe_size:
                break
            x = _to_device(x, device_t)
            logits_chunks.append(model(x).cpu())
            seen += x.shape[0] if isinstance(x, torch.Tensor) else x["input_ids"].shape[0]
    logits = torch.cat(logits_chunks, dim=0)[:probe_size]

    # DP noise: clip per-example logit L2 to 1, then Gaussian mechanism.
    logits = logits / logits.norm(dim=-1, keepdim=True).clamp(min=1.0)
    sigma_dp = gaussian_mechanism_sigma(1.0, eps, delta)
    logits_dp = logits + torch.randn_like(logits) * sigma_dp

    return {
        "type": "logits",
        "logits": logits_dp.numpy(),
        "eps": eps,
        "delta": delta,
    }


def summarise_synthetic(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str,
    eps: float,
    delta: float,
    n_synthetic: int = 128,
) -> dict[str, Any]:
    """DP synthetic microbatch.

    Placeholder: draws n_synthetic samples with random labels; noise is added to
    the raw pixel/token tensor. Replace with DP-MERF or similar for production.
    """
    xs = []
    ys = []
    for x, y in _iter_batches(dataset, batch_size=32):
        xs.append(x if isinstance(x, torch.Tensor) else x["input_ids"])
        ys.append(torch.as_tensor(y))
        if sum(z.shape[0] for z in xs) >= n_synthetic:
            break
    x_cat = torch.cat(xs, dim=0)[:n_synthetic].float()
    y_cat = torch.cat(ys, dim=0)[:n_synthetic].long()
    # Add DP noise to a normalised x.
    x_cat = x_cat / x_cat.abs().flatten(1).max(dim=-1, keepdim=True).values.clamp(min=1e-6).view(-1, *([1] * (x_cat.ndim - 1)))
    sigma_dp = gaussian_mechanism_sigma(1.0, eps, delta)
    x_dp = x_cat + torch.randn_like(x_cat) * sigma_dp
    return {
        "type": "synthetic",
        "xs": x_dp.numpy(),
        "ys": y_cat.numpy(),
        "eps": eps,
        "delta": delta,
    }


def summarise_sketch(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str,
    eps: float,
    delta: float,
    sketch_dim: int = 512,
) -> dict[str, Any]:
    """Sketched gradient summary.

    Concatenates one epoch of gradients into a random projection of dimension
    `sketch_dim` and adds DP noise. Expected to underperform the other three
    on memorisation-heavy tasks.
    """
    model.train()
    device_t = torch.device(device)
    loss_fn = nn.CrossEntropyLoss()
    total = None
    n = 0
    for x, y in _iter_batches(dataset, batch_size=32):
        x = _to_device(x, device_t)
        y = torch.as_tensor(y, device=device_t)
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        # Flatten grads.
        g = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
        total = g if total is None else total + g
        n += 1
    if total is None:
        return {"type": "sketch", "sketch": np.zeros(sketch_dim), "eps": eps, "delta": delta}
    g_avg = total / max(n, 1)
    torch.manual_seed(0)
    P = torch.randn(sketch_dim, g_avg.numel(), device=g_avg.device) / np.sqrt(sketch_dim)
    sketch = P @ g_avg
    sketch = sketch / sketch.norm().clamp(min=1.0)
    sigma_dp = gaussian_mechanism_sigma(1.0, eps, delta)
    sketch_dp = sketch + torch.randn_like(sketch) * sigma_dp
    return {
        "type": "sketch",
        "sketch": sketch_dp.detach().cpu().numpy(),
        "eps": eps,
        "delta": delta,
    }


SUMMARISERS = {
    "moments": summarise_moments,
    "logits": summarise_logits,
    "synthetic": summarise_synthetic,
    "sketch": summarise_sketch,
}


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)
