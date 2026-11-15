"""Train a federated continual learning model with the decayed-aggregation protocol.

Usage:
    python scripts/train.py --config configs/cifar100_combined.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

# Allow `python scripts/train.py` from the code/ root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.datasets import build_dataset
from src.federated import DecayedAggregator, Server
from src.federated.aggregator import DecayState
from src.federated.loop import cpu_state_dict, federated_train
from src.federated.replay import ReplayBuffer
from src.federated.backdoor import BackdoorConfig
from src.metrics import average_test_accuracy
from src.models import build_model
from src.unlearning import SUMMARISERS
from src.utils import load_config, set_seed, setup_logger, resolve_device


def _dataset_kwargs(cfg) -> dict:
    ds = cfg.section("dataset").to_dict()
    root = cfg["experiment"]["data_root"]
    if ds["name"] in ("cifar10", "cifar100"):
        return {"data_root": root, "num_clients": ds["num_clients"], "num_tasks": ds["num_tasks"],
                "dirichlet_alpha": ds["dirichlet_alpha"], "seed": cfg["experiment"]["seed"]}
    if ds["name"] == "tiny_imagenet":
        return {"data_root": root, "num_clients": ds["num_clients"], "num_tasks": ds["num_tasks"],
                "dirichlet_alpha": ds["dirichlet_alpha"], "seed": cfg["experiment"]["seed"]}
    if ds["name"] == "femnist":
        return {"data_root": root, "num_clients": ds.get("num_clients"), "seed": cfg["experiment"]["seed"]}
    if ds["name"] == "newsgroups20":
        return {"data_root": root, "num_clients": ds["num_clients"], "num_tasks": ds["num_tasks"],
                "dirichlet_alpha": ds["dirichlet_alpha"], "tokenizer_name": ds["tokenizer_name"],
                "max_len": ds["max_len"], "seed": cfg["experiment"]["seed"]}
    raise ValueError(f"Unsupported dataset in config: {ds['name']}")


def _model_kwargs(cfg, num_classes: int) -> dict:
    m = cfg.section("model").to_dict()
    if m["name"] == "resnet18":
        return {"num_classes": num_classes, "in_channels": m.get("in_channels", 3),
                "small_input": m.get("small_input", False)}
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("fu", log_dir=ckpt_dir)
    logger.info(f"Config: {args.config}")

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    logger.info(f"Dataset: {fcl_split.metadata} clients={fcl_split.num_clients} tasks={fcl_split.num_tasks}")

    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def model_factory():
        return build_model(cfg["model"]["name"], **model_kwargs)

    # Init global state.
    initial_model = model_factory().to(exp["device"])
    global_state = cpu_state_dict(initial_model)

    # Server + aggregator.
    agg = DecayedAggregator(
        alpha=cfg["aggregator"]["alpha"],
        server_lr=cfg["aggregator"]["server_lr"],
        server_sigma=cfg["aggregator"]["server_sigma"],
        seed=exp["seed"],
    )
    server = Server(
        global_state=global_state,
        decay_state=DecayState.zeros_from(global_state, alpha=cfg["aggregator"]["alpha"]),
        aggregator=agg,
    )

    # Summary channel.
    summarise_fn = None
    if cfg["summary"]["enabled"]:
        stype = cfg["summary"]["type"]
        if stype not in SUMMARISERS:
            raise ValueError(f"Unknown summary type: {stype}")
        summarise_fn = SUMMARISERS[stype]

    # Eval callback.
    def evaluate(server: Server, fcl_split, task_id: int) -> dict:
        model = model_factory().to(exp["device"])
        model.load_state_dict(server.global_state)
        acc = average_test_accuracy(model, fcl_split, device=exp["device"])
        return {"avg_test_acc": acc}

    rep_cfg = cfg.section("replay")
    replay_buffer = None
    if rep_cfg.get("enabled", False):
        replay_buffer = ReplayBuffer(
            capacity_per_client=int(rep_cfg.get("capacity_per_client", 20)),
            admit_fraction=float(rep_cfg.get("admit_fraction", 0.5)),
            seed=exp["seed"],
        )

    bd_cfg = cfg.section("backdoor")
    backdoor = None
    if bd_cfg.get("enabled", False):
        backdoor = BackdoorConfig(
            client_id=int(bd_cfg.get("client_id", 3)),
            rounds_before_erasure=int(bd_cfg.get("rounds_before_erasure", 20)),
            target_label=int(bd_cfg.get("target_label", 0)),
            patch_size=int(bd_cfg.get("patch_size", 4)),
            patch_value=float(bd_cfg.get("patch_value", 2.5)),
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
        backdoor=backdoor,
    )

    torch.save(
        {"global_state": server.global_state, "summaries": server.summaries,
         "decay_state": server.decay_state.tilde_delta, "decay_alpha": server.decay_state.alpha},
        ckpt_dir / "final.pt",
    )
    logger.info(f"Training complete. Final checkpoint: {ckpt_dir / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
