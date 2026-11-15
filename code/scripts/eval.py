"""Evaluate a pre/post erasure checkpoint pair on all metric families.

Usage:
    python scripts/eval.py \
        --pre runs/cifar100_combined/ckpt_r200.pt \
        --post runs/cifar100_combined/ckpt_r200_unlearned_c42.pt \
        --client 42 \
        --config configs/cifar100_combined.yaml
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

from src.attacks import mia_yeom, mia_lira_lite
from src.datasets import build_dataset
from src.metrics import average_test_accuracy
from src.metrics.forgetting import retrain_kl
from src.models import build_model
from src.utils import load_config, set_seed, setup_logger, resolve_device


def _dataset_kwargs(cfg) -> dict:
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


def _model_kwargs(cfg, num_classes: int) -> dict:
    m = cfg.section("model").to_dict()
    if m["name"] == "resnet18":
        return {"num_classes": num_classes, "in_channels": m.get("in_channels", 3), "small_input": m.get("small_input", False)}
    if m["name"] == "small_cnn":
        return {"num_classes": num_classes, "in_channels": m.get("in_channels", 1)}
    if m["name"] == "tiny_cnn":
        return {"num_classes": num_classes, "in_channels": m.get("in_channels", 3),
                "input_size": m.get("input_size", 32)}
    if m["name"] == "text_transformer":
        return {"num_classes": num_classes, "model_name": m.get("model_name", "distilbert-base-uncased")}
    raise ValueError(f"Unsupported model: {m['name']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--client", type=int, required=True)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))
    logger = setup_logger("fu-eval")

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def load_model(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = build_model(cfg["model"]["name"], **model_kwargs).to(exp["device"])
        m.load_state_dict(ckpt["global_state"])
        return m, ckpt

    m_pre, ckpt_pre = load_model(args.pre)
    m_post, ckpt_post = load_model(args.post)

    # Utility (average test accuracy across all tasks).
    acc_pre = average_test_accuracy(m_pre, fcl_split, device=exp["device"])
    acc_post = average_test_accuracy(m_post, fcl_split, device=exp["device"])

    # MIA over the client's data (members) vs held-out non-member samples.
    client_idx = fcl_split.client_indices[args.client].tolist()
    member_set = Subset(fcl_split.train_set, client_idx)
    # Non-members: random subset of other clients' data, matched size.
    all_other = np.concatenate(
        [fcl_split.client_indices[i] for i in range(fcl_split.num_clients) if i != args.client]
    ).tolist()
    rng = np.random.default_rng(exp["seed"])
    nonmember_idx = rng.choice(all_other, size=min(len(client_idx), len(all_other)), replace=False).tolist()
    nonmember_set = Subset(fcl_split.train_set, nonmember_idx)

    mia_pre_y = mia_yeom(m_pre, member_set, nonmember_set, device=exp["device"])
    mia_post_y = mia_yeom(m_post, member_set, nonmember_set, device=exp["device"])
    mia_pre_l = mia_lira_lite(m_pre, member_set, nonmember_set, device=exp["device"])
    mia_post_l = mia_lira_lite(m_post, member_set, nonmember_set, device=exp["device"])

    # KL between post-erasure and pre-erasure over a probe. (For proper retrain-KL,
    # substitute a retrain-from-scratch model in place of pre.)
    from torch.utils.data import Subset as _S
    probe = _S(fcl_split.test_set, list(range(min(512, len(fcl_split.test_set)))))
    kl = retrain_kl(m_post, m_pre, probe, device=exp["device"])

    results = {
        "utility": {"acc_pre": acc_pre, "acc_post": acc_post, "delta": acc_post - acc_pre},
        "mia_yeom_pre": mia_pre_y,
        "mia_yeom_post": mia_post_y,
        "mia_lira_lite_pre": mia_pre_l,
        "mia_lira_lite_post": mia_post_l,
        "retrain_kl_post_vs_pre": kl,
        "certificate": ckpt_post.get("certificate", None),
    }

    out = json.dumps(results, indent=2, default=str)
    logger.info(out)
    if args.output_json:
        Path(args.output_json).write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
