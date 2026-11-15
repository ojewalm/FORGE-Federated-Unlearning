"""Train a single FedAvg-with-history base that all five baselines reuse.

Produces:
    <output>/final_base.pt      — {'global_state', 'round_index', 'participation'}
    <output>/history.pt         — list of {'round': r, 'deltas': {cid: delta}}
    <output>/initial_state.pt   — theta_0 for FedEraser
    <output>/ckpt_r*.pt         — periodic numbered checkpoints for SIFU
    <output>/resume.pt          — running snapshot for interruptable resume

Usage:
    python scripts/train_shared_base.py --config configs/cifar100_real.yaml \
        --output runs/cifar100_real/shared_base
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.baselines.common import train_fedavg_and_maybe_store_history
from src.datasets import build_dataset
from src.metrics import average_test_accuracy
from src.models import build_model
from src.utils import load_config, resolve_device, set_seed, setup_logger


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
        return {"num_classes": n, "in_channels": m.get("in_channels", 3), "input_size": m.get("input_size", 32)}
    if m["name"] == "text_transformer":
        return {"num_classes": n, "model_name": m.get("model_name", "distilbert-base-uncased")}
    raise ValueError(m["name"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("fu-shared-base", log_dir=out)
    logger.info(f"Training shared FedAvg base with history. Config: {args.config}")

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def model_factory():
        return build_model(cfg["model"]["name"], **model_kwargs)

    # theta_0 snapshot for FedEraser.
    init_model = model_factory().to(exp["device"])
    init_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in init_model.state_dict().items())
    torch.save(init_state, out / "initial_state.pt")

    def evaluate(server, fcl_split, task_id: int) -> dict:
        model = model_factory().to(exp["device"])
        model.load_state_dict(server.global_state)
        return {"avg_test_acc": average_test_accuracy(model, fcl_split, device=exp["device"])}

    result = train_fedavg_and_maybe_store_history(
        model_factory=lambda: model_factory().to(exp["device"]),
        fcl_split=fcl_split,
        total_rounds=cfg["federated"]["total_rounds"],
        rounds_per_task=cfg["federated"]["rounds_per_task"],
        client_sample_frac=cfg["federated"]["client_sample_frac"],
        local_epochs=cfg["federated"]["local_epochs"],
        batch_size=cfg["federated"]["batch_size"],
        lr=cfg["federated"]["lr"],
        clip_norm=cfg["federated"]["clip_norm"],
        device=exp["device"],
        checkpoint_dir=out,
        seed=exp["seed"],
        store_history=True,     # always store: all history-based baselines need it
        history_stride=cfg.section("baselines").get("history_stride", 5),
        evaluate_every=cfg["evaluation"]["evaluate_every"],
        evaluate_fn=evaluate,
        checkpoint_every=cfg.section("evaluation").get("checkpoint_every", 1),
        resume=True,
    )

    torch.save({
        "global_state": result["global_state"],
        "round_index": result["round_index"],
        "participation": result["participation"],
    }, out / "final_base.pt")
    # History is already on disk as per-round files under `out/history/`.
    # train_fedavg_and_maybe_store_history returns a `history_dir` pointer
    # rather than loading the (potentially multi-GB) list into memory.
    n_history_files = len(list((out / "history").glob("r*.pt"))) if (out / "history").exists() else 0
    logger.info(f"Shared base saved to {out}. Rounds={result['round_index']} "
                f"history_entries={n_history_files} (per-round files under history/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
