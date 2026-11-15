"""Decayed aggregation (Algorithm 1)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from .client import ClientUpdate
from .params import (
    ParamDict,
    add_state,
    scale_state,
    zero_state_like,
    add_gaussian_noise,
)


@dataclass
class DecayState:
    """Running aggregation state: `tilde_delta` from Eq. (2)."""
    tilde_delta: ParamDict
    alpha: float

    @classmethod
    def zeros_from(cls, model_state: ParamDict, alpha: float) -> "DecayState":
        return cls(tilde_delta=zero_state_like(model_state), alpha=alpha)


class DecayedAggregator:
    """Feature-decoupled decayed aggregation.

    Implements Algorithm 1 of the paper: the feature block (keys not starting
    with any `head_key_prefix`) is updated with (i) EMA decay α on the running
    aggregation state, and (ii) per-parameter Gaussian noise of std σ. The
    head block (keys starting with any `head_key_prefix`) is updated with
    unweighted FedAvg and no server noise.

    This mechanical split is what makes Theorem 3's parallel composition
    bound apply: decay and summary-based subtraction then act on disjoint
    parameter blocks, so the overall privacy budget is max(ε₁, ε₂) rather
    than ε₁ + ε₂.

    For legacy FedAvg behaviour (no decay, no noise, full-state EMA with
    α ≈ 0), set `alpha` close to 0 and `server_sigma=0`.
    """

    DEFAULT_HEAD_PREFIXES = ("head.",)

    def __init__(
        self,
        alpha: float,
        server_lr: float,
        server_sigma: float,
        seed: int = 0,
        head_key_prefixes: tuple[str, ...] = DEFAULT_HEAD_PREFIXES,
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1).")
        # alpha == 0 reduces the decay recursion to vanilla FedAvg (no EMA).
        self.alpha = alpha
        self.server_lr = server_lr
        self.server_sigma = server_sigma
        self.head_key_prefixes = head_key_prefixes
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def _is_head_key(self, key: str) -> bool:
        return any(key.startswith(p) for p in self.head_key_prefixes)

    def aggregate(
        self,
        global_state: ParamDict,
        updates: list[ClientUpdate],
        decay_state: DecayState,
    ) -> tuple[ParamDict, DecayState]:
        if not updates:
            return global_state, decay_state

        # Sample-weighted mean of client deltas this round (full state).
        total = sum(u.num_samples for u in updates)
        avg = zero_state_like(global_state)
        for u in updates:
            w = u.num_samples / total
            avg = add_state(avg, u.delta, scale=w)

        # Per-key split: decay+noise on features; unweighted FedAvg on head.
        # Integer buffers (BN's num_batches_tracked) bypass decay and noise
        # since they are not meaningfully "parameters" for the DP analysis.
        new_tilde = OrderedDict()
        new_global = OrderedDict()
        for k, v_global in global_state.items():
            if not v_global.is_floating_point():
                # Pass-through for integer buffers: take the max over clients.
                new_tilde[k] = avg[k].clone()
                new_global[k] = v_global + avg[k].to(v_global.dtype)
            elif self._is_head_key(k):
                # Head: no decay, no noise.
                new_tilde[k] = avg[k].clone()
                new_global[k] = v_global + self.server_lr * avg[k]
            else:
                # Feature: EMA decay + Gaussian noise.
                tilde_prev = decay_state.tilde_delta[k]
                tilde_new = self.alpha * tilde_prev + (1.0 - self.alpha) * avg[k]
                new_tilde[k] = tilde_new
                noise = torch.randn_like(v_global) * self.server_sigma
                new_global[k] = v_global + self.server_lr * (tilde_new + noise)

        return new_global, DecayState(tilde_delta=new_tilde, alpha=self.alpha)
