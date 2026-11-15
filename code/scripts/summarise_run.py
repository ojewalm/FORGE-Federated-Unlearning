"""Aggregate the outputs of a train / unlearn / retrain / eval sequence into one row.

Usage:
    python scripts/summarise_run.py \
        --pre  runs/cifar100_smoke/final.pt \
        --post runs/cifar100_smoke/final_unlearned_c3.pt \
        --retrain runs/cifar100_smoke/retrain_excl_c3/final.pt \
        --config configs/cifar100_smoke.yaml \
        --client 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Subset

from src.attacks import mia_yeom
from src.datasets import build_dataset
from src.metrics import average_test_accuracy
from src.metrics.forgetting import retrain_kl
from src.models import build_model
from src.utils import load_config, set_seed, resolve_device


def _dataset_kwargs(cfg):
    ds = cfg.section("dataset").to_dict()
    root = cfg["experiment"]["data_root"]
    if ds["name"] in ("cifar10", "cifar100", "tiny_imagenet"):
        return {"data_root": root, "num_clients": ds["num_clients"], "num_tasks": ds["num_tasks"],
                "dirichlet_alpha": ds["dirichlet_alpha"], "seed": cfg["experiment"]["seed"]}
    if ds["name"] == "femnist":
        return {"data_root": root, "num_clients": ds.get("num_clients"), "seed": cfg["experiment"]["seed"]}
    if ds["name"] == "newsgroups20":
        return {"data_root": root, "num_clients": ds["num_clients"], "num_tasks": ds["num_tasks"],
                "dirichlet_alpha": ds["dirichlet_alpha"], "tokenizer_name": ds["tokenizer_name"],
                "max_len": ds["max_len"], "seed": cfg["experiment"]["seed"]}
    raise ValueError(f"Unsupported dataset: {ds['name']}")


def _model_kwargs(cfg, n):
    m = cfg.section("model").to_dict()
    if m["name"] == "resnet18":
        return {"num_classes": n, "in_channels": m.get("in_channels", 3), "small_input": m.get("small_input", False)}
    if m["name"] == "small_cnn":
        return {"num_classes": n, "in_channels": m.get("in_channels", 1)}
    if m["name"] == "tiny_cnn":
        return {"num_classes": n, "in_channels": m.get("in_channels", 3),
                "input_size": m.get("input_size", 32)}
    if m["name"] == "text_transformer":
        return {"num_classes": n, "model_name": m.get("model_name", "distilbert-base-uncased")}
    raise ValueError(f"Unsupported model: {m['name']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--retrain", default=None)
    ap.add_argument("--client", type=int, required=True)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def load_ckpt(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(cfg["model"]["name"], **model_kwargs).to(exp["device"])
        m.load_state_dict(ck["global_state"])
        return m, ck

    m_pre, ck_pre = load_ckpt(args.pre)
    m_post, ck_post = load_ckpt(args.post)
    m_retrain, ck_retrain = (load_ckpt(args.retrain) if args.retrain else (None, None))

    # Utility.
    acc_pre = average_test_accuracy(m_pre, fcl_split, device=exp["device"])
    acc_post = average_test_accuracy(m_post, fcl_split, device=exp["device"])
    acc_retrain = average_test_accuracy(m_retrain, fcl_split, device=exp["device"]) if m_retrain is not None else None

    # MIA on the departed client's data vs a non-member subset.
    client_idx = fcl_split.client_indices[args.client].tolist()
    if not client_idx:
        return _fail("Client has no data; cannot compute MIA.")
    member = Subset(fcl_split.train_set, client_idx)
    others = np.concatenate([fcl_split.client_indices[i] for i in range(fcl_split.num_clients) if i != args.client]).tolist()
    rng = np.random.default_rng(exp["seed"])
    non_idx = rng.choice(others, size=min(len(client_idx), len(others)), replace=False).tolist()
    non = Subset(fcl_split.train_set, non_idx)

    mia_pre = mia_yeom(m_pre, member, non, device=exp["device"])
    mia_post = mia_yeom(m_post, member, non, device=exp["device"])
    mia_retrain = mia_yeom(m_retrain, member, non, device=exp["device"]) if m_retrain is not None else None

    # Retrain-KL (post vs retrain if we have it; else post vs pre for reference).
    probe = Subset(fcl_split.test_set, list(range(min(256, len(fcl_split.test_set)))))
    kl_post_vs_retrain = retrain_kl(m_post, m_retrain, probe, device=exp["device"]) if m_retrain is not None else None
    kl_post_vs_pre = retrain_kl(m_post, m_pre, probe, device=exp["device"])

    row = {
        "client": args.client,
        "config": args.config,
        "acc_pre": round(acc_pre, 4),
        "acc_post": round(acc_post, 4),
        "acc_retrain": round(acc_retrain, 4) if acc_retrain is not None else None,
        "acc_delta_post_vs_pre": round(acc_post - acc_pre, 4),
        "acc_delta_post_vs_retrain": round(acc_post - acc_retrain, 4) if acc_retrain is not None else None,
        "mia_adv_pre": round(mia_pre["advantage"], 4),
        "mia_adv_post": round(mia_post["advantage"], 4),
        "mia_adv_retrain": round(mia_retrain["advantage"], 4) if mia_retrain is not None else None,
        "mia_adv_drop_post_vs_pre": round(mia_pre["advantage"] - mia_post["advantage"], 4),
        "kl_post_vs_retrain": round(kl_post_vs_retrain, 4) if kl_post_vs_retrain is not None else None,
        "kl_post_vs_pre": round(kl_post_vs_pre, 4),
        "certificate": ck_post.get("certificate"),
    }

    out = json.dumps(row, indent=2, default=str)
    print(out)
    if args.output_json:
        Path(args.output_json).write_text(out)
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
