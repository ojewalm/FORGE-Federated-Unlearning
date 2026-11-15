"""Empirical Lipschitz probe for the LocalTrain map at ResNet-18 scale.

For a saved global state theta0 and a client dataset D_i, we measure

    L' ~= ||LocalTrain(theta0 + eps*v, D_i) - LocalTrain(theta0, D_i)||_2 / eps

with deterministic batch shuffling across both runs so that only the input-state
perturbation is responsible for the output deviation. Averaging across clients
and random perturbation directions gives an empirical estimate of the per-step
Lipschitz constant L' that appears in the contraction condition
eta_g (1 - alpha) eta_l L' < 1 of Appendix A.4.
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.datasets import build_dataset
from src.federated.client import Client
from src.models import build_model
from src.utils import load_config, resolve_device, set_seed


def _is_trainable_param(key: str) -> bool:
    # Exclude BatchNorm running_mean, running_var, num_batches_tracked (auxiliary stats, not params).
    return not any(suffix in key for suffix in ("running_mean", "running_var", "num_batches_tracked"))


def _state_norm(state: dict) -> float:
    sq = 0.0
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if v.dtype in (torch.int64, torch.int32, torch.long):
            continue
        if not _is_trainable_param(k):
            continue
        sq += float((v.float() ** 2).sum().item())
    return float(sq) ** 0.5


def _state_sub(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if torch.is_tensor(a[k]) and torch.is_tensor(b[k]) and a[k].dtype == b[k].dtype and a[k].dtype.is_floating_point:
            out[k] = a[k] - b[k]
        else:
            out[k] = a[k]
    return out


def _state_add(a: dict, b: dict, scale: float = 1.0) -> dict:
    out = {}
    for k in a:
        if torch.is_tensor(a[k]) and a[k].dtype.is_floating_point:
            out[k] = a[k] + scale * b[k]
        else:
            out[k] = a[k]
    return out


def _rand_direction(reference: dict, rng: np.random.Generator) -> dict:
    v = {}
    sq = 0.0
    for k, p in reference.items():
        if torch.is_tensor(p) and p.dtype.is_floating_point and _is_trainable_param(k):
            arr = rng.standard_normal(p.shape).astype(np.float32)
            v[k] = torch.from_numpy(arr)
            sq += float((v[k] ** 2).sum().item())
    norm = sq ** 0.5
    for k in v:
        v[k] = v[k] / norm
    for k, p in reference.items():
        if k not in v:
            v[k] = torch.zeros_like(p) if torch.is_tensor(p) else p
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--n-directions", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device("cpu")  # cpu for reproducibility of local training

    ds = cfg.section("dataset").to_dict()
    name = ds.pop("name")
    ds["data_root"] = cfg["experiment"]["data_root"]
    ds["seed"] = cfg["experiment"]["seed"]
    fcl = build_dataset(name, **ds)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    theta0 = checkpoint.get("global_state", checkpoint)

    model_cfg = cfg.section("model").to_dict()
    model_cfg.setdefault("num_classes", fcl.num_classes)

    def build():
        return build_model(name=model_cfg["name"], num_classes=fcl.num_classes,
                           in_channels=model_cfg.get("in_channels", 3),
                           small_input=model_cfg.get("small_input", True))

    rng = np.random.default_rng(args.seed)

    local_epochs = int(cfg["federated"]["local_epochs"])
    batch_size = int(cfg["federated"]["batch_size"])
    lr = float(cfg["federated"]["lr"])
    clip_norm = float(cfg["federated"]["clip_norm"])
    eps = args.epsilon

    chosen_cids = rng.choice(fcl.num_clients, size=min(args.n_clients, fcl.num_clients), replace=False)

    all_L = []
    for cid in chosen_cids:
        client_ds = fcl.client_dataset(int(cid), task_id=fcl.num_tasks - 1)
        if client_ds is None or len(client_ds) == 0:
            continue
        client = Client(client_id=int(cid), dataset=client_ds,
                        local_epochs=local_epochs, batch_size=batch_size,
                        lr=lr, clip_norm=clip_norm, device=device)

        # Baseline run with deterministic seed.
        set_seed(1234)
        random.seed(1234)
        torch.manual_seed(1234)
        model = build().to(device)
        upd_base = client.local_update(theta0, model, round_index=199)

        # Reconstruct baseline local_state = theta0 + delta (unclipped-equivalent view).
        base_local_state = {k: theta0[k].cpu().float() + upd_base.delta[k].float()
                            if (torch.is_tensor(theta0[k]) and torch.is_tensor(upd_base.delta.get(k)) and theta0[k].dtype.is_floating_point)
                            else theta0[k]
                            for k in theta0}

        for d in range(args.n_directions):
            v = _rand_direction(theta0, rng)
            theta_p = _state_add(theta0, v, scale=eps)

            set_seed(1234)
            random.seed(1234)
            torch.manual_seed(1234)
            model2 = build().to(device)
            upd_p = client.local_update(theta_p, model2, round_index=199)
            pert_local_state = {k: theta_p[k].cpu().float() + upd_p.delta[k].float()
                                if (torch.is_tensor(theta_p[k]) and torch.is_tensor(upd_p.delta.get(k)) and theta_p[k].dtype.is_floating_point)
                                else theta_p[k]
                                for k in theta_p}

            delta_out = _state_sub(pert_local_state, base_local_state)
            # exclude non-floating keys
            delta_out = {k: v for k, v in delta_out.items() if torch.is_tensor(v) and v.dtype.is_floating_point}
            num = _state_norm(delta_out)
            L = num / eps
            all_L.append(L)
            print(f"client={int(cid)} dir={d} ||delta_out||={num:.4f} L'={L:.3f}")

    if not all_L:
        print("no measurements")
        return 1
    arr = np.array(all_L)
    print(f"\n=== L' summary ({len(arr)} measurements) ===")
    print(f"mean   : {arr.mean():.3f}")
    print(f"median : {np.median(arr):.3f}")
    print(f"min    : {arr.min():.3f}")
    print(f"max    : {arr.max():.3f}")
    print(f"p95    : {np.quantile(arr, 0.95):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
