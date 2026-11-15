"""Train the same config while excluding a target client — gold-standard retrain baseline.

Usage:
    python scripts/retrain.py --config configs/cifar100_smoke.yaml --exclude-client 3
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.datasets import build_dataset
from src.federated import DecayedAggregator, Server
from src.federated.aggregator import DecayState
from src.federated.loop import cpu_state_dict, federated_train
from src.federated.replay import ReplayBuffer
from src.metrics import average_test_accuracy
from src.models import build_model
from src.unlearning import SUMMARISERS
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
    ap.add_argument("--exclude-client", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))

    # Force a distinct output dir so we don't stomp the main run.
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"]) / f"retrain_excl_c{args.exclude_client}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("fu-retrain", log_dir=ckpt_dir)
    logger.info(f"Retrain excluding client {args.exclude_client}. Config: {args.config}")

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    # Zero out the excluded client's indices so they never participate.
    fcl_split.client_indices[args.exclude_client] = np.array([], dtype=np.int64)

    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def model_factory():
        return build_model(cfg["model"]["name"], **model_kwargs)

    initial_model = model_factory().to(exp["device"])
    global_state = cpu_state_dict(initial_model)

    # Retrain-from-scratch is a gold-standard baseline: it reflects what a
    # standard FL protocol produces without any unlearning procedure. We
    # therefore use vanilla FedAvg (alpha≈0, no server noise), not our
    # method's decayed + noised aggregator. Comparison stays fair because
    # the five baseline unlearners also run on a vanilla FedAvg base.
    agg = DecayedAggregator(alpha=1e-6, server_lr=1.0, server_sigma=0.0, seed=exp["seed"])
    server = Server(
        global_state=global_state,
        decay_state=DecayState.zeros_from(global_state, alpha=1e-6),
        aggregator=agg,
    )

    # No summary channel in the gold-retrain either.
    summarise_fn = None

    def evaluate(server: Server, fcl_split, task_id: int) -> dict:
        model = model_factory().to(exp["device"])
        model.load_state_dict(server.global_state)
        return {"avg_test_acc": average_test_accuracy(model, fcl_split, device=exp["device"])}

    rep_cfg = cfg.section("replay")
    replay_buffer = None
    if rep_cfg.get("enabled", False):
        replay_buffer = ReplayBuffer(
            capacity_per_client=int(rep_cfg.get("capacity_per_client", 20)),
            admit_fraction=float(rep_cfg.get("admit_fraction", 0.5)),
            seed=exp["seed"],
        )

    federated_train(
        server=server,
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
        summarise_fn=summarise_fn,
        summary_frequency=cfg["summary"]["frequency"],
        summary_eps=cfg["summary"]["eps"],
        summary_delta=cfg["summary"]["delta"],
        evaluate_every=cfg["evaluation"]["evaluate_every"],
        evaluate_fn=evaluate,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=cfg.section("evaluation").get("checkpoint_every", 1),
        seed=exp["seed"],
        dataset_name=cfg["dataset"]["name"],
        replay_buffer=replay_buffer,
    )

    torch.save(
        {"global_state": server.global_state, "summaries": server.summaries,
         "decay_state": server.decay_state.tilde_delta, "decay_alpha": server.decay_state.alpha,
         "excluded_client": args.exclude_client},
        ckpt_dir / "final.pt",
    )
    logger.info(f"Retrain complete. Checkpoint: {ckpt_dir / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
