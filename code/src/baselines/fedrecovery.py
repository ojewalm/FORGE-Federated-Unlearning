"""Zhang et al. (2023) — FedRecovery.

Paper: Lefeng Zhang, Tianqing Zhu, Haibin Zhang, Ping Xiong, Wanlei Zhou.
"FedRecovery: Differentially Private Machine Unlearning for Federated Learning
Frameworks." IEEE TIFS 18 (2023), pp. 4732–4746.

No public reference code exists (verified via GitHub/Papers-with-Code/author
pages as of 2026-09-30). This implementation follows the algorithm in the
paper.

Method
------
At training time, the server retains a *weighted running sum* of each client's
contributions. When an erasure request arrives for client i*, the server:

  1. Retrieves the stored contribution history H_{i*} = { (r, Δ_{i*}^{(r)}) }.
  2. Computes a *residual direction* d = Σ_r w_r · Δ_{i*}^{(r)}, weighted by
     the client's participation weight at each round.
  3. Subtracts a scaled version of d from the final global model:
     θ ← θ − η · d.
  4. Adds calibrated Gaussian noise ξ ~ N(0, σ² I) to the result to guarantee
     that the recovered model is (ε, δ)-indistinguishable from a retrain
     without client i*.

The paper's ε, δ analysis calibrates σ = C · sqrt(2 ln(1.25/δ)) / ε, where C
bounds the L2 norm of the aggregated recovery direction after clipping.

Notes
-----
- This baseline is the closest DP-guarantee competitor to our combined
  protocol. Its weakness (which our method targets) is that it requires
  storing per-round update history, i.e. O(N × R × |θ|) at the server.
- The paper's before/after model pair is exactly what enables the Chen-et-al
  MIA-on-unlearning and FUIA attacks — so this baseline's `pre` and `post`
  models should both be evaluated by FUIA in the write-up.
"""
from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn

from ..federated.params import ParamDict, add_state, clip_state, scale_state, state_norm, zero_state_like
from ..utils.dp import gaussian_mechanism_sigma
from .common import BaselineResult


logger = logging.getLogger("fu")


def fedrecovery_unlearn(
    *,
    global_state: ParamDict,
    history: list[dict],           # list of {'round': r, 'deltas': {cid: delta}}
    forget_client_id: int,
    weighting: str = "linear",     # "linear" (recent rounds weighted more) or "uniform"
    eta: float = 1.0,              # step size for the residual subtraction
    dp_eps: float = 1.0,
    dp_delta: float = 1e-5,
    dp_sensitivity: float = 1.0,   # L2 bound on aggregated direction
) -> BaselineResult:
    """FedRecovery unlearning.

    Args:
        global_state: final trained model to recover from.
        history: per-round update deltas, indexed by client_id, from the
                 history-storing training run.
        forget_client_id: departing client's index.
        weighting: how to weight historical contributions. Linear places more
            weight on later rounds (matching the paper's default). Uniform is
            the naive baseline.
        eta: step size.
        dp_eps, dp_delta: DP budget for the calibrated noise.
        dp_sensitivity: assumed L2 bound on the aggregated recovery direction
            (must be enforced by clipping). Set this equal to the per-round
            clip constant C times a reasonable participation bound.
    """
    tic = time.time()

    # Filter history to just the target client's contributions.
    contributions = [(entry["round"], entry["deltas"][forget_client_id])
                     for entry in history if forget_client_id in entry["deltas"]]
    if not contributions:
        # Nothing to subtract; still emit a certificate for completeness.
        logger.warning(f"FedRecovery: no stored contributions for client {forget_client_id}")
        return BaselineResult(
            method="fedrecovery",
            unlearned_state=OrderedDict((k, v.clone()) for k, v in global_state.items()),
            wall_time_s=time.time() - tic,
            guarantee={"eps": dp_eps, "delta": dp_delta},
            extras={"num_contributions_subtracted": 0, "weighting": weighting},
        )

    # Compute per-round weights.
    max_r = max(r for r, _ in contributions)
    if weighting == "linear":
        weights = {r: (r + 1) / (max_r + 1) for r, _ in contributions}
    elif weighting == "uniform":
        weights = {r: 1.0 for r, _ in contributions}
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    wsum = sum(weights.values())

    # Aggregate weighted residual direction.
    d = zero_state_like(global_state)
    for r, delta in contributions:
        d = add_state(d, delta, scale=weights[r] / wsum)

    # Clip to enforce the assumed sensitivity.
    d = clip_state(d, dp_sensitivity)

    # Subtract, then add Gaussian noise calibrated to (eps, delta).
    # dp_eps=math.inf (float('inf')) is interpreted as the no-noise anchor
    # (requested by reviewers to anchor the destroy-at-every-tested-eps claim).
    unlearned = add_state(global_state, d, scale=-eta)
    if math.isinf(dp_eps):
        sigma = 0.0
        noisy = OrderedDict((k, v.clone()) for k, v in unlearned.items())
    else:
        sigma = gaussian_mechanism_sigma(dp_sensitivity, dp_eps, dp_delta)
        noisy = OrderedDict()
        for k, v in unlearned.items():
            if v.is_floating_point():
                noisy[k] = v + torch.randn_like(v) * sigma
            else:
                noisy[k] = v.clone()

    return BaselineResult(
        method="fedrecovery",
        unlearned_state=noisy,
        wall_time_s=time.time() - tic,
        guarantee={"eps": dp_eps, "delta": dp_delta},
        extras={
            "num_contributions_subtracted": len(contributions),
            "weighting": weighting,
            "eta": eta,
            "sigma": sigma,
            "sensitivity": dp_sensitivity,
            "requires_history_storage": True,
        },
    )
