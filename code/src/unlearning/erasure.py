"""Non-interactive erasure procedure (Algorithm 2)."""
from __future__ import annotations

import hashlib
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..federated.params import ParamDict, add_state, clip_state
from ..federated.server import CommittedSummary, Server
from ..utils.dp import forward_forgetting_epsilon, composed_budget


@dataclass
class ErasureCertificate:
    client_id: int
    r_forget: int
    r_request: int
    eps_1: float
    delta_1: float
    eps_2: float
    delta_2: float
    eps_total: float
    delta_total: float
    parallel_composition: bool
    summary_commitment: str
    transcript_hash: str
    wall_time_s: float
    method: str = "combined"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _kl_between_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    log_pa = F.log_softmax(a, dim=-1)
    log_pb = F.log_softmax(b, dim=-1)
    p_a = log_pa.exp()
    return (p_a * (log_pa - log_pb)).sum(dim=-1).mean()


def _residual_direction_from_logits(
    model: nn.Module,
    probe_logits_dp: torch.Tensor,
    probe_input: torch.Tensor | dict,
    head_only: bool,
) -> ParamDict:
    """Compute grad of KL(model || summary) w.r.t. parameters. This points
    towards the client's contributions; we will subtract along it."""
    model.zero_grad()
    if isinstance(probe_input, dict):
        current = model(probe_input)
    else:
        current = model(probe_input)
    # The summary logits were computed on the client's local data at summary
    # release time, and may have fewer samples than the current probe batch
    # (e.g., clients with small Dirichlet shards). Truncate both tensors to
    # their common row count before computing KL.
    n = min(current.shape[0], probe_logits_dp.shape[0])
    loss = _kl_between_logits(current[:n], probe_logits_dp[:n])
    if head_only:
        params = list(model.head_parameters())
    else:
        params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    out = OrderedDict()
    named = dict(model.named_parameters())
    grad_map = {id(p): g for p, g in zip(params, grads)}
    for name, p in named.items():
        if id(p) in grad_map and grad_map[id(p)] is not None:
            out[name] = grad_map[id(p)].detach().cpu()
        else:
            out[name] = torch.zeros_like(p).cpu()
    # Cover buffers (non-trainable) with zeros so add_state alignment holds.
    full = OrderedDict(model.state_dict())
    for k in full:
        if k not in out:
            out[k] = torch.zeros_like(full[k]).cpu()
    return out


def _residual_direction_from_moments(
    model: nn.Module,
    summary_payload: dict,
    device: torch.device,
    head_only: bool,
) -> ParamDict:
    """Push the classifier head *away from* the client's per-class feature means
    by treating each per-class mean as a virtual example.
    """
    means = summary_payload["means"]
    if len(means) == 0:
        return OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in model.state_dict().items())
    xs = torch.tensor(np.stack(list(means.values())), dtype=torch.float32, device=device)
    ys = torch.tensor(list(means.keys()), dtype=torch.long, device=device)
    if hasattr(model, "logits_from_features"):
        logits = model.logits_from_features(xs)
    else:
        logits = model(xs)
    loss = F.cross_entropy(logits, ys)
    if head_only:
        params = list(model.head_parameters())
    else:
        params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    named = dict(model.named_parameters())
    grad_map = {id(p): g for p, g in zip(params, grads)}
    out = OrderedDict()
    for name, p in named.items():
        if id(p) in grad_map and grad_map[id(p)] is not None:
            out[name] = grad_map[id(p)].detach().cpu()
        else:
            out[name] = torch.zeros_like(p).cpu()
    for k in model.state_dict():
        if k not in out:
            out[k] = torch.zeros_like(model.state_dict()[k]).cpu()
    return out


def non_interactive_erasure(
    *,
    server: Server,
    model_factory,                        # () -> nn.Module
    client_id: int,
    r_forget: int,
    r_request: int,
    forward_forgetting_params: dict,     # alpha, C, sigma, rho, delta_1
    ascent_step: float = 0.1,
    projection_radius: float = 0.5,
    head_only: bool = True,               # enables parallel composition
    device: str = "cpu",
    probe_input: torch.Tensor | dict | None = None,
) -> tuple[ParamDict, ErasureCertificate]:
    """Algorithm 2: apply the non-interactive erasure procedure."""
    tic = time.time()
    summary = server.get_summary(client_id)
    if summary is None:
        raise ValueError(f"No stored summary for client {client_id}; erasure requires pre-commitment.")

    # Verify commitment.
    payload_bytes = pickle.dumps(summary.payload)
    h = hashlib.sha256()
    h.update(int(client_id).to_bytes(8, "little"))
    h.update(int(summary.round_index).to_bytes(8, "little"))
    h.update(payload_bytes)
    if h.hexdigest() != summary.commitment:
        raise ValueError("Summary commitment mismatch — possible tampering.")

    # Compute residual direction and apply bounded ascent step.
    device_t = torch.device(device)
    model = model_factory().to(device_t)
    model.load_state_dict(server.global_state)
    stype = summary.summary_type
    if stype == "logits":
        logits_dp = torch.tensor(summary.payload["logits"], dtype=torch.float32, device=device_t)
        if probe_input is None:
            raise ValueError("probe_input is required for logits-based erasure.")
        direction = _residual_direction_from_logits(model, logits_dp, probe_input, head_only=head_only)
    elif stype == "moments":
        direction = _residual_direction_from_moments(model, summary.payload, device_t, head_only=head_only)
    else:
        # Sketch / synthetic: fall back to a zero step and pass through decay-only guarantee.
        direction = OrderedDict((k, torch.zeros_like(v).cpu()) for k, v in server.global_state.items())

    # Project ascent step to radius R.
    direction_clipped = clip_state(direction, projection_radius)
    new_global = add_state(server.global_state, direction_clipped, scale=ascent_step)

    # Compute the certificate.
    T = r_request - r_forget
    eps_1 = forward_forgetting_epsilon(
        T=T,
        alpha=forward_forgetting_params["alpha"],
        clip_C=forward_forgetting_params["clip_C"],
        sigma=forward_forgetting_params["sigma"],
        rho=forward_forgetting_params["rho"],
        delta=forward_forgetting_params["delta_1"],
    )
    eps_2, delta_2 = summary.eps, summary.delta
    eps_total, delta_total = composed_budget(
        eps_1, eps_2, forward_forgetting_params["delta_1"], delta_2, parallel=head_only
    )
    transcript = {
        "client_id": client_id,
        "r_forget": r_forget,
        "r_request": r_request,
        "commitment": summary.commitment,
        "eps_1": eps_1,
        "eps_2": eps_2,
        "params": forward_forgetting_params,
        "head_only": head_only,
    }
    transcript_bytes = pickle.dumps(transcript)
    transcript_hash = hashlib.sha256(transcript_bytes).hexdigest()

    cert = ErasureCertificate(
        client_id=client_id,
        r_forget=r_forget,
        r_request=r_request,
        eps_1=eps_1,
        delta_1=forward_forgetting_params["delta_1"],
        eps_2=eps_2,
        delta_2=delta_2,
        eps_total=eps_total,
        delta_total=delta_total,
        parallel_composition=head_only,
        summary_commitment=summary.commitment,
        transcript_hash=transcript_hash,
        wall_time_s=time.time() - tic,
        method="combined",
    )

    # Update server, purge summary.
    server.global_state = new_global
    server.purge_summary(client_id)

    return new_global, cert
