"""Wei et al. (KDD 2026) — memorisation-scoped federated unlearning.

Paper: "Rethinking Federated Unlearning via the Lens of Memorization." KDD 2026.
arXiv:2605.24545.

Reference code:
https://github.com/JWei1999/Rethinking-Federated-Unlearning-via-the-Lens-of-Memorization
(MIT). Cross-checked against the reference notebooks.

Method
------
Identify parameters that carry *memorised* content specific to the departing
client, and reset (or dampen) only those parameters. Overlapping structure
supported by other clients is preserved. The paper's core signal is:

    memorisation_score[p] = |grad_{f(model)} L_client_i(p)|  (approximately)

evaluated on the target client's data, then normalised across parameters.
Parameters above a quantile threshold are re-initialised (or scaled by a
small factor); the rest are left intact. Optional short fine-tuning follows.

This baseline is the closest scope competitor to our summary-based subtraction
step: both are non-history-based and non-interactive-at-server-side, but the
memorisation mask has no formal (ε, δ) guarantee.
"""
from __future__ import annotations

import copy
import logging
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..federated.params import ParamDict
from .common import BaselineResult


logger = logging.getLogger("fu")


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def _memorisation_score(
    model: nn.Module,
    forget_dataset: Dataset,
    device: str,
    batch_size: int = 32,
    max_batches: int = 20,
) -> dict[str, torch.Tensor]:
    """Compute per-parameter |grad| L2 magnitude on the client's data.

    This approximates each parameter's marginal contribution to fitting the
    client's data, as in the paper's Sec. 3.2 (approximate influence via
    gradient magnitude). We accumulate over up to `max_batches` batches to
    keep this cheap.
    """
    model.eval()
    model.zero_grad()
    accum: dict[str, torch.Tensor] = {}
    loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = _to_device(x, device)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(
            loss, [p for p in model.parameters() if p.requires_grad], retain_graph=False, allow_unused=True
        )
        named = [n for n, p in model.named_parameters() if p.requires_grad]
        for n, g in zip(named, grads):
            if g is None:
                continue
            v = g.detach().abs()
            accum[n] = v if n not in accum else accum[n] + v
    return accum


def fedmemprune_unlearn(
    *,
    global_state: ParamDict,
    initial_state: ParamDict,
    model_factory: Callable[[], nn.Module],
    forget_dataset: Dataset,
    device: str = "cpu",
    prune_quantile: float = 0.95,
    reset_scale: float = 0.0,        # 0 => reinitialise to initial_state; e.g. 0.5 => dampen
    include_head: bool = True,
) -> BaselineResult:
    """Reset the top-`(1 - prune_quantile)` most-memorising parameters.

    Args:
        prune_quantile: e.g. 0.95 → prune the top 5% of parameters ranked by
            per-parameter |grad| magnitude on the client's data.
        reset_scale: 0.0 replaces those entries with `initial_state`
            (equivalent to reinitialisation at the pre-training state). Larger
            values do a soft dampening `theta <- (1 - s) * theta_init + s * theta`.
        include_head: whether to prune classifier-head parameters too. The
            paper prunes across the whole model; the head-only variant is our
            ablation.
    """
    tic = time.time()

    model = model_factory()
    model.load_state_dict(global_state)
    model.to(device)
    scores = _memorisation_score(model, forget_dataset, device)

    # Flatten scores to pick a global quantile. Move to CPU because
    # torch.kthvalue has no MPS kernel as of torch 2.8.
    concat = torch.cat([s.flatten().cpu() for s in scores.values()]) if scores else torch.tensor([])
    if concat.numel() == 0:
        threshold = float("inf")
    else:
        k = int(prune_quantile * concat.numel())
        if k > 0:
            th_tensor, _ = torch.kthvalue(concat, k)
            threshold = float(th_tensor.item())
        else:
            threshold = float(concat.min().item())

    # Apply mask.
    head_prefixes = ("head.",) if not include_head else ()
    new_state = OrderedDict()
    for name, tensor in global_state.items():
        if name in scores:
            mask = (scores[name].cpu() > threshold).float()
            if any(name.startswith(p) for p in head_prefixes):
                new_state[name] = tensor  # skip head
                continue
            init_v = initial_state[name] if name in initial_state else torch.zeros_like(tensor)
            keep = (1.0 - mask)
            new_v = keep * tensor + mask * ((1 - reset_scale) * init_v + reset_scale * tensor)
            new_state[name] = new_v
        else:
            new_state[name] = tensor

    return BaselineResult(
        method="fedmemprune",
        unlearned_state=new_state,
        wall_time_s=time.time() - tic,
        guarantee={},   # heuristic; no formal indistinguishability bound
        extras={
            "prune_quantile": prune_quantile,
            "reset_scale": reset_scale,
            "threshold_value": threshold,
            "num_masked_params": int(sum(((scores[n].cpu() > threshold).sum() for n in scores))) if scores else 0,
        },
    )
