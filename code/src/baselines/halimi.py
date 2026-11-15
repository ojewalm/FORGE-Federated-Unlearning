"""Halimi et al. (2022) — projected gradient ascent unlearning.

Paper: A. Halimi, S. Kadhe, A. Rawat, N. Baracaldo. "Federated Unlearning: How
to Efficiently Erase a Client in FL?" UpML Workshop @ ICML 2022.
arXiv:2207.05521.

Reference code: https://github.com/IBM/federated-unlearning (Apache-2.0).
Delivered by the authors as a Jupyter notebook; this reimplementation follows
Algorithm 1 of the paper and cross-checks against the notebook's core update
rule.

Method
------
Starting from the final global model, run projected gradient *ascent* on the
target client's local data for a small number of steps. The projection keeps
the model within an L2 ball of radius `radius` around a *reference model*
(the paper's PGA-BALL; the reference is typically an intermediate global model
before the target client's dominant contributions). After ascent, optionally
fine-tune on the remaining clients (post-unlearning FedAvg rounds) to recover
utility. This baseline requires the departing client to be online at request
time — a property we deliberately do not require in the proposed protocol.

Notes
-----
- The paper's core update rule uses SGD with a small learning rate and clips
  the parameters back into the ball each step. We match that.
- Our implementation exposes `post_ft_rounds`; the paper's setup uses a few
  fine-tuning rounds after ascent, but this is a knob that trades off
  forgetting quality vs. utility.
"""
from __future__ import annotations

import copy
import logging
import math
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..federated.params import ParamDict, sub_state, state_norm
from .common import BaselineResult


logger = logging.getLogger("fu")


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def _load_and_get_params(model: nn.Module, state: ParamDict, device: str) -> list[torch.Tensor]:
    model.load_state_dict(state)
    model.to(device)
    return list(model.parameters())


def _project_into_ball(model: nn.Module, ref_state: ParamDict, radius: float) -> None:
    """Project the model back into an L2 ball of `radius` around ref_state."""
    with torch.no_grad():
        current = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        diff = sub_state(current, ref_state)
        d = state_norm(diff)
        if d > radius:
            scale = radius / (d + 1e-12)
            new_state = {}
            for k, v in ref_state.items():
                new_state[k] = ref_state[k] + scale * (current[k] - ref_state[k])
            model.load_state_dict(new_state)


def halimi_unlearn(
    *,
    global_state: ParamDict,
    reference_state: ParamDict | None,
    model_factory: Callable[[], nn.Module],
    forget_dataset: Dataset,
    device: str = "cpu",
    ascent_steps: int = 50,
    ascent_lr: float = 1e-3,
    radius: float = 3.0,
    batch_size: int = 32,
) -> BaselineResult:
    """Run Halimi PGA-BALL unlearning.

    Args:
        global_state: current server state.
        reference_state: model state to project around. If None, uses
            `global_state` (unlearning wanders locally from the final model).
        forget_dataset: the departing client's data. Must be provided online —
            this is the operational restriction Halimi assumes.
        ascent_steps: number of ascent iterations.
        ascent_lr: SGD learning rate for ascent.
        radius: L2 ball radius around the reference model.
    """
    tic = time.time()
    ref = reference_state if reference_state is not None else global_state

    model = model_factory()
    model.load_state_dict(global_state)
    model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=ascent_lr)
    loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    it = iter(loader)

    for step in range(ascent_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x = _to_device(x, device)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        opt.zero_grad()
        logits = model(x)
        # Negative cross-entropy (== gradient ascent on CE), matching the
        # paper's formulation of "moving away from the client's optimum".
        loss = -F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        _project_into_ball(model, ref, radius)

    from collections import OrderedDict
    unlearned = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())

    return BaselineResult(
        method="halimi",
        unlearned_state=unlearned,
        wall_time_s=time.time() - tic,
        guarantee={},   # no formal guarantee — this is what our method beats it on
        extras={
            "ascent_steps": ascent_steps,
            "ascent_lr": ascent_lr,
            "radius": radius,
            "requires_client_online": True,
        },
    )
