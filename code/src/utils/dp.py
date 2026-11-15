"""Differential privacy primitives.

Implements Gaussian mechanism and (basic) RDP accounting for composition of
the client-side summary channel and server-side noise. Kept intentionally
lightweight; production experiments should route through Opacus for gradient
DP-SGD noise, and this module for the summary channel and forward-forgetting
rate calculation.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def gaussian_noise(shape: tuple[int, ...], sigma: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """Standard Gaussian noise with std sigma."""
    return torch.randn(*shape, device=device) * sigma


def gaussian_mechanism_sigma(sensitivity: float, eps: float, delta: float) -> float:
    """Sigma for the (eps, delta)-Gaussian mechanism (analytic Gaussian, simple form).

    For a query of L2 sensitivity `sensitivity`, adding N(0, sigma^2 I) noise
    yields (eps, delta)-DP when sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / eps.
    This is the standard (loose) bound; tighten with the analytic Gaussian
    mechanism for reporting.
    """
    if eps <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("Require eps > 0 and delta in (0, 1).")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps


def forward_forgetting_epsilon(
    T: int,
    alpha: float,
    clip_C: float,
    sigma: float,
    rho: float,
    delta: float,
) -> float:
    """Epsilon(T) for the forward-forgetting theorem (Theorem 1 of the paper).

    eps_1(T) = C * sqrt(2 * rho * T * log(1/delta)) * alpha^T / (sigma * (1 - alpha))

    This is the certified rate for the aggregation-state residual under the
    decayed-aggregation protocol.
    """
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0, 1).")
    if T < 0 or clip_C <= 0 or sigma < 0 or rho <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("Invalid parameters for forward-forgetting bound.")
    # alpha = 0 reduces the aggregation state to vanilla FedAvg; the
    # forward-forgetting rate collapses to zero (no decay-based decay) and the
    # overall guarantee is supplied entirely by the summary-subtraction cost.
    if alpha == 0.0 or sigma == 0.0:
        return math.inf
    num = clip_C * math.sqrt(2.0 * rho * max(T, 1) * math.log(1.0 / delta))
    return num * (alpha ** T) / (sigma * (1.0 - alpha))


def cooling_off_time(
    eps_target: float,
    alpha: float,
    clip_C: float,
    sigma: float,
    rho: float,
    delta: float,
    T_max: int = 10_000,
) -> int:
    """Smallest T at which eps_1(T) <= eps_target. Returns T_max if not reached."""
    for T in range(1, T_max):
        if forward_forgetting_epsilon(T, alpha, clip_C, sigma, rho, delta) <= eps_target:
            return T
    return T_max


def composed_budget(eps1: float, eps2: float, delta1: float, delta2: float, parallel: bool = False) -> tuple[float, float]:
    """Compose two DP budgets. Sequential by default; parallel when disjoint subspace."""
    if parallel:
        return max(eps1, eps2), max(delta1, delta2)
    return eps1 + eps2, delta1 + delta2


def clip_l2(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale `tensor` so its L2 norm is at most `max_norm`. In-place friendly."""
    norm = tensor.norm(p=2)
    if norm > max_norm:
        tensor = tensor * (max_norm / (norm + 1e-12))
    return tensor
