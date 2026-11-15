"""Apply the non-interactive erasure procedure to a checkpoint.

Usage:
    python scripts/unlearn.py --checkpoint runs/cifar100_combined/ckpt_r200.pt \
        --client 42 --r-forget 100 --r-request 200 --config configs/cifar100_combined.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.datasets import build_dataset
from src.federated import DecayedAggregator, Server
from src.federated.aggregator import DecayState
from src.models import build_model
from src.unlearning import non_interactive_erasure
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--client", type=int, required=True)
    ap.add_argument("--r-forget", type=int, required=True)
    ap.add_argument("--r-request", type=int, required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))
    logger = setup_logger("fu-unlearn")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def model_factory():
        return build_model(cfg["model"]["name"], **model_kwargs)

    agg = DecayedAggregator(
        alpha=cfg["aggregator"]["alpha"],
        server_lr=cfg["aggregator"]["server_lr"],
        server_sigma=cfg["aggregator"]["server_sigma"],
        seed=exp["seed"],
    )
    server = Server(
        global_state=ckpt["global_state"],
        decay_state=DecayState(tilde_delta=ckpt.get("decay_state", OrderedDict()), alpha=ckpt.get("decay_alpha", cfg["aggregator"]["alpha"])),
        aggregator=agg,
        summaries=ckpt.get("summaries", {}),
        round_index=args.r_request,
    )

    # Prepare probe input for logits-based erasure (a small batch from fcl_split.test_set).
    from torch.utils.data import DataLoader
    probe_loader = DataLoader(fcl_split.test_set, batch_size=64, shuffle=False)
    x_probe, _ = next(iter(probe_loader))
    if isinstance(x_probe, dict):
        x_probe = {k: v.to(exp["device"]) for k, v in x_probe.items()}
    else:
        x_probe = x_probe.to(exp["device"])

    ff = cfg["unlearning"]["forward_forgetting"]
    _, cert = non_interactive_erasure(
        server=server,
        model_factory=lambda: model_factory().to(exp["device"]),
        client_id=args.client,
        r_forget=args.r_forget,
        r_request=args.r_request,
        forward_forgetting_params={
            "alpha": cfg["aggregator"]["alpha"],
            "clip_C": cfg["federated"]["clip_norm"],
            "sigma": cfg["aggregator"]["server_sigma"],
            "rho": ff["rho"],
            "delta_1": ff["delta_1"],
        },
        ascent_step=cfg["unlearning"]["ascent_step"],
        projection_radius=cfg["unlearning"]["projection_radius"],
        head_only=cfg["unlearning"]["head_only"],
        device=exp["device"],
        probe_input=x_probe,
    )

    out_path = Path(args.output) if args.output else Path(args.checkpoint).with_name(
        Path(args.checkpoint).stem + f"_unlearned_c{args.client}.pt"
    )
    torch.save(
        {"global_state": server.global_state,
         "summaries": server.summaries,
         "decay_state": server.decay_state.tilde_delta,
         "decay_alpha": server.decay_state.alpha,
         "certificate": cert.as_dict()},
        out_path,
    )
    logger.info(f"Unlearned checkpoint saved: {out_path}")
    logger.info(f"Certificate: {json.dumps(cert.as_dict(), indent=2, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
