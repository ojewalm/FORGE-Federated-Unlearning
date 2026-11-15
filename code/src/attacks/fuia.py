"""Federated Unlearning Inversion Attack (FUIA) — reconstruction via
pre/post model-pair gradient inversion.

Reference: Zhou \\& Zhu, "FUIA: Model Inversion Attack against Federated
Unlearning", arXiv:2502.14558.

The attack premise: an unlearning procedure releases both a pre-erasure
model theta_pre and a post-erasure model theta_post. The difference
theta_pre - theta_post encodes the gradient signal of the departing
client's data. An adversary with (theta_pre, theta_post) solves for a
sample x such that the model's gradient on (x, y_target) best explains the
observed difference.

Our implementation follows the inversion-via-optimisation framework of
Geiping et al. (2020), adapted to the federated-unlearning setting: start
with a random x, use Adam to minimise the cosine distance between the
observed delta = theta_pre - theta_post and the actual per-sample gradient
grad_x = nabla loss(theta_pre, x, y_target).

This is a non-stub implementation suitable for running in the paper's
evaluation. It is substantially slower than the attack stub it replaces
(~1-10 minutes per reconstruction attempt) and should be run against a
representative set of samples (default: 32 per method).

The attack is a stronger threat for history-releasing unlearners
(FedRecovery, FedEraser) that necessarily publish a pre/post pair. For
FORGE, the pre-erasure model is never published; verification uses the
certificate's hash commitments. The attack cannot be mounted against a
single released post-erasure model, so FORGE's reported FUIA quality is
measured against the no-unlearning baseline (post == pre).
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("fu")


def _flatten_grads(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grads if g is not None])


def fuia_reconstruct_sample(
    model_pre: nn.Module,
    model_post: nn.Module,
    target_label: int,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    steps: int = 500,
    lr: float = 0.1,
    tv_weight: float = 1e-4,
    verbose: bool = False,
) -> torch.Tensor:
    """Reconstruct a single sample of class `target_label`.

    Minimises cosine distance between the observed parameter delta
    (theta_pre - theta_post) and the per-sample gradient computed at
    model_pre for (x, target_label), plus a small total-variation
    regulariser on x. Returns the reconstructed x on CPU.
    """
    device_t = torch.device(device)
    model_pre.eval()
    model_post.eval()
    # Compute the observed parameter delta (what the unlearning erased).
    delta_flat = _flatten_grads(
        [p_pre.detach().to(device_t) - p_post.detach().to(device_t)
         for p_pre, p_post in zip(model_pre.parameters(), model_post.parameters())]
    )
    delta_norm = delta_flat.norm().clamp(min=1e-8)

    # Initialise x in [-1, 1]^input_shape.
    x = torch.randn((1,) + tuple(input_shape), device=device_t, requires_grad=True)
    y = torch.tensor([target_label], device=device_t)
    opt = torch.optim.Adam([x], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_cos, best_x = -1.0, None

    for step in range(steps):
        opt.zero_grad()
        model_pre.zero_grad()
        logits = model_pre(x)
        ce = loss_fn(logits, y)
        grads = torch.autograd.grad(ce, list(model_pre.parameters()), create_graph=True, allow_unused=True)
        g_flat = _flatten_grads([g for g in grads if g is not None])
        g_norm = g_flat.norm().clamp(min=1e-8)
        cos = (g_flat * delta_flat).sum() / (g_norm * delta_norm)
        # Total variation regulariser (penalises high-frequency noise).
        if x.ndim == 4:  # image
            tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + \
                 (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        else:
            tv = torch.tensor(0.0, device=device_t)
        objective = -(cos) + tv_weight * tv
        objective.backward()
        opt.step()
        if cos.item() > best_cos:
            best_cos = cos.item()
            best_x = x.detach().clone()
        if verbose and (step + 1) % 100 == 0:
            logger.info(f"[fuia] step {step+1}/{steps} cos={cos.item():.4f}")

    return best_x.detach().cpu() if best_x is not None else x.detach().cpu()


def fuia_attack(
    *,
    model_pre: nn.Module,
    model_post: nn.Module,
    forget_dataset,
    num_classes: int,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    n_samples: int = 16,
    steps: int = 500,
    lr: float = 0.1,
) -> dict:
    """Run FUIA on n_samples drawn from the forget set, report quality.

    For each sampled true (x, y), reconstruct x_hat from (model_pre,
    model_post) using y as the target label. Report mean PSNR and cosine
    similarity between x and x_hat.

    The attack is more effective when the pre/post pair encodes a strong
    signal from the forgotten client (i.e., when the unlearning method
    actually changed the model). For an unlearning method that only
    performs a bounded parameter change (e.g., FORGE's head-only
    projection), the signal is weak and reconstruction quality is poor.
    """
    from torch.utils.data import DataLoader
    model_pre.to(device).eval()
    model_post.to(device).eval()

    psnr_list, cos_list = [], []
    loader = DataLoader(forget_dataset, batch_size=1, shuffle=True)
    for i, (x_true, y_true) in enumerate(loader):
        if i >= n_samples:
            break
        y_int = int(y_true.item() if torch.is_tensor(y_true) else y_true)
        try:
            x_rec = fuia_reconstruct_sample(
                model_pre, model_post, target_label=y_int,
                input_shape=input_shape, device=device, steps=steps, lr=lr,
            )
        except RuntimeError as e:
            # Some models (e.g. small_cnn's MaxPool2d) do not support the
            # second-order gradients FUIA requires. Report NaN gracefully so
            # the benchmark can continue to the next method.
            logger.warning(f"[fuia] reconstruction failed: {e}. Skipping.")
            return {"n_samples": n_samples, "mean_psnr": float("nan"),
                    "mean_cosine_sim": float("nan"),
                    "std_psnr": 0.0, "std_cosine_sim": 0.0,
                    "error": str(e)[:200]}
        q = _reconstruction_quality(x_true.cpu().flatten(), x_rec.cpu().flatten())
        psnr_list.append(q["psnr"])
        cos_list.append(q["cosine_sim"])
    return {
        "n_samples": n_samples,
        "mean_psnr": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "mean_cosine_sim": float(np.mean(cos_list)) if cos_list else float("nan"),
        "std_psnr": float(np.std(psnr_list)) if psnr_list else 0.0,
        "std_cosine_sim": float(np.std(cos_list)) if cos_list else 0.0,
    }


def _reconstruction_quality(x_true: torch.Tensor, x_rec: torch.Tensor) -> dict:
    """PSNR + cosine similarity. Inputs are flattened CPU tensors."""
    x_true = x_true.float()
    x_rec = x_rec.float()
    mse = float(((x_true - x_rec) ** 2).mean())
    peak = max(float(x_true.abs().max()), 1e-6)
    psnr = 10 * np.log10(peak ** 2 / max(mse, 1e-12))
    cos = float(F.cosine_similarity(x_true.unsqueeze(0), x_rec.unsqueeze(0)).item())
    return {"mse": mse, "psnr": psnr, "cosine_sim": cos}


# Keep the old function name for backward compatibility.
def fuia_reconstruct(model_pre, model_post, input_shape, num_classes, target_label,
                     device="cpu", steps=500, lr=0.1):
    return fuia_reconstruct_sample(
        model_pre=model_pre, model_post=model_post, target_label=target_label,
        input_shape=input_shape, device=device, steps=steps, lr=lr,
    )
