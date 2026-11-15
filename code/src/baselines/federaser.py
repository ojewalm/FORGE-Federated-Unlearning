"""Liu et al. (2021) — FedEraser.

Paper: Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, Jiangchuan Liu.
"FedEraser: Enabling Efficient Client-Level Data Removal from Federated
Learning Models." IWQoS 2021.

Reference mirror: https://github.com/Zoybzo/FedEraser-Code — no LICENSE file,
so we cannot legally vendor. Reimplemented from the paper description.

Method
------
At training time, the server retains per-round per-client updates for all
clients. At unlearning time:

  1. Restart from a small number of "calibration" rounds before the departing
     client first participated (or from `theta_0`).
  2. Replay the stored updates *without* the departing client, but instead of
     recomputing them, use each retained update as an approximation of the
     direction the server would have gone in that round.
  3. To calibrate for the fact that the retained updates were produced with
     the departing client's influence still in the model, scale each retained
     update by a small factor and re-orient in the direction of the current
     retained-updates-only mean.

Concretely, at each retained round r:
    d_bar_r = mean_{i != i*} Δ_i^{(r)}
    theta_r <- theta_{r-1} + lr_calib * d_bar_r

`lr_calib` is a small step size (the paper uses ~0.5 * server_lr). The final
recovered model is the trajectory endpoint.

Notes
-----
- Requires history storage during training (same as FedRecovery, SIFU).
- No formal (ε, δ) guarantee; empirical MIA reduction is the reported
  benefit in the paper.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn

from ..federated.params import ParamDict, add_state, scale_state, zero_state_like
from .common import BaselineResult


logger = logging.getLogger("fu")


def federaser_unlearn(
    *,
    initial_state: ParamDict,
    history: list[dict],
    forget_client_id: int,
    lr_calib: float = 0.5,
) -> BaselineResult:
    """Replay retained-only mean updates from the initial state.

    Args:
        initial_state: model state at the beginning of training (theta_0).
        history: [{'round': r, 'deltas': {cid: delta}}, ...].
        forget_client_id: client to omit.
        lr_calib: multiplicative calibration for each replay step.
    """
    tic = time.time()

    theta = OrderedDict((k, v.clone()) for k, v in initial_state.items())
    replayed_rounds = 0

    for entry in history:
        deltas_wo_forget = {cid: d for cid, d in entry["deltas"].items() if cid != forget_client_id}
        if not deltas_wo_forget:
            continue
        n = len(deltas_wo_forget)
        d_bar = zero_state_like(theta)
        for cid, d in deltas_wo_forget.items():
            d_bar = add_state(d_bar, d, scale=1.0 / n)
        theta = add_state(theta, d_bar, scale=lr_calib)
        replayed_rounds += 1

    return BaselineResult(
        method="federaser",
        unlearned_state=theta,
        wall_time_s=time.time() - tic,
        guarantee={},
        extras={
            "replayed_rounds": replayed_rounds,
            "lr_calib": lr_calib,
            "requires_history_storage": True,
        },
    )
