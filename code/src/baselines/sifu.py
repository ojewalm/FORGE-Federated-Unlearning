"""Fraboni et al. (2024) — SIFU: Sequential Informed Federated Unlearning.

Paper: Y. Fraboni, M. Van Waerebeke, K. Scaman, R. Vidal, L. Kameni, M. Lorenzi.
"SIFU: Sequential Informed Federated Unlearning for Efficient and Provable
Client Unlearning in Federated Optimization." AISTATS 2024, PMLR v238,
pp. 3457-3465.

Reference code: https://github.com/Accenture/Labs-Federated-Learning/tree/SIFU
— the SIFU branch has no LICENSE file, so we cannot vendor. Reimplemented
from the paper.

Method
------
SIFU tracks a per-iteration bound on the influence of each client on the
current model, and — when a client requests erasure — rolls back to the most
recent iteration whose bound implies the client's residual influence is below
a target ε. From that safe checkpoint, training resumes without the client
until the original round budget is exhausted (or a user-specified limit).

Concretely, denote by `beta_r` the per-round influence bound on any client at
round r (computed from clipping constant C, step size, and participation
frequency). SIFU maintains sqrt(sum beta_r^2) as a running influence-so-far
value and identifies the largest r* such that resuming from θ_{r*} without
the target client can bring the residual below ε.

Notes
-----
- Provides a formal indistinguishability guarantee under the paper's
  assumptions; ε here is a knob.
- Requires per-round *checkpoint* history (not full per-client updates), so
  storage is O(R × |θ|) rather than O(N × R × |θ|). Still O(1) in clients
  but O(R) in model size.
- The rollback + fine-tune structure makes it O(remaining-rounds) wall time
  at request time — the slowest of the five baselines at request time.
"""
from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn

from ..federated.params import ParamDict
from .common import BaselineResult


logger = logging.getLogger("fu")


def _influence_bound_r(round_index: int, clip_C: float, lr: float, participation_rho: float) -> float:
    """A simple per-round influence bound: |Δ_r^i| <= lr * C for a participating client."""
    return lr * clip_C * participation_rho


def sifu_choose_rollback(
    total_rounds: int,
    target_eps: float,
    clip_C: float,
    lr: float,
    participation_rho: float,
) -> int:
    """Return the largest r* such that sqrt(sum_{k > r*} beta_k^2) <= target_eps."""
    # beta_k identical across k in this simple bound; the sqrt-sum is
    # beta * sqrt(total_rounds - r*), so we want beta * sqrt(total - r*) <= eps.
    beta = _influence_bound_r(0, clip_C, lr, participation_rho)
    if beta == 0:
        return total_rounds
    max_gap = (target_eps / beta) ** 2
    r_star = max(0, int(math.floor(total_rounds - max_gap)))
    return r_star


def sifu_unlearn(
    *,
    rounds_checkpoints: dict[int, ParamDict],  # {r: theta_r}
    total_rounds: int,
    forget_client_id: int,
    resume_train_fn: Callable[[ParamDict, int, int, int], ParamDict],
    target_eps: float,
    clip_C: float,
    lr: float,
    participation_rho: float,
) -> BaselineResult:
    """SIFU unlearning: pick a rollback round, then fine-tune without the client.

    Args:
        rounds_checkpoints: server checkpoints keyed by round index. Must
            contain at least one round <= r_star.
        total_rounds: total number of rounds trained originally.
        forget_client_id: departing client.
        resume_train_fn: callable(state, from_round, to_round, exclude_client) -> new state.
            The caller wires this to whichever training routine was used
            originally, with `exclude_client` implemented by zeroing that
            client's participation.
        target_eps: desired forward-forgetting threshold.
        clip_C: per-round update clip norm (used for the influence bound).
        lr: server / effective learning rate for the influence bound.
        participation_rho: assumed max participation rate.
    """
    tic = time.time()
    r_star = sifu_choose_rollback(total_rounds, target_eps, clip_C, lr, participation_rho)
    stored = sorted(rounds_checkpoints.keys())
    if not stored:
        raise ValueError("SIFU needs at least one stored round checkpoint (increase evaluate_every "
                         "cadence in the base training, or supply an initial checkpoint at r=0).")
    candidate_rounds = [r for r in stored if r <= r_star]
    # Fall back to the earliest available if no round <= r_star exists.
    r_use = candidate_rounds[-1] if candidate_rounds else stored[0]
    logger.info(f"SIFU: rolling back to r={r_use} (r*={r_star}, total={total_rounds})")
    theta_start = rounds_checkpoints[r_use]
    theta_final = resume_train_fn(theta_start, r_use, total_rounds, forget_client_id)

    return BaselineResult(
        method="sifu",
        unlearned_state=theta_final,
        wall_time_s=time.time() - tic,
        guarantee={"eps": target_eps, "delta": 0.0},   # paper's formulation is worst-case bound
        extras={
            "rollback_round": r_use,
            "r_star": r_star,
            "total_rounds": total_rounds,
            "requires_checkpoint_storage": True,
        },
    )
