"""Train a FedAvg base and apply one of the five baseline unlearning methods.

Usage:
    python scripts/run_baseline.py --config configs/cifar100_smoke.yaml \
        --method halimi --forget-client 3

Methods: halimi, fedmemprune, fedrecovery, federaser, sifu.

For methods that need history (fedrecovery, federaser, sifu), history is
stored during the FedAvg base training. For the others (halimi, fedmemprune),
we still store round-numbered checkpoints so SIFU can also be run against the
same base training run without re-training.

Interruptible: base training is resumable via the same snapshot mechanism the
main training loop uses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Subset

from src.baselines import (
    BASELINES,
    federaser_unlearn,
    fedmemprune_unlearn,
    fedrecovery_unlearn,
    halimi_unlearn,
    sifu_choose_rollback,
    sifu_unlearn,
    train_fedavg_and_maybe_store_history,
)
from src.datasets import build_dataset
from src.metrics import average_test_accuracy
from src.models import build_model
from src.utils import load_config, resolve_device, set_seed, setup_logger


HISTORY_METHODS = {"federaser", "fedrecovery", "sifu"}


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
    ap.add_argument("--method", required=True, choices=sorted(BASELINES))
    ap.add_argument("--forget-client", type=int, required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--shared-base", default=None,
                    help="Path to a shared base directory containing final_base.pt "
                         "(and, for history methods, history.pt). If provided, training "
                         "is skipped and this base is reused.")
    ap.add_argument("--fedrecovery-eps", type=float, default=None,
                    help="Override FedRecovery's DP eps budget (otherwise uses config value). "
                         "Used for the eps-sweep requested by reviewers.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))

    # Allow per-eps subdirs so the FedRecovery eps-sweep doesn't overwrite itself.
    tag = f"baseline_{args.method}_c{args.forget_client}"
    if args.method == "fedrecovery" and args.fedrecovery_eps is not None:
        eps_str = "inf" if args.fedrecovery_eps == float("inf") else str(args.fedrecovery_eps)
        tag += f"_eps{eps_str}"
    ckpt_root = Path(cfg["output"]["checkpoint_dir"]) / tag
    ckpt_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"fu-{args.method}", log_dir=ckpt_root)
    logger.info(f"Baseline {args.method} for client {args.forget_client}. Config: {args.config}")

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def model_factory():
        return build_model(cfg["model"]["name"], **model_kwargs)

    def evaluate(server, fcl_split, task_id: int) -> dict:
        model = model_factory().to(exp["device"])
        model.load_state_dict(server.global_state)
        return {"avg_test_acc": average_test_accuracy(model, fcl_split, device=exp["device"])}

    # ---- Base training (or reuse shared base) ----------------------------
    initial_state = None
    if args.method == "federaser":
        # FedEraser needs theta_0; capture it before training.
        init_model = model_factory().to(exp["device"])
        initial_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in init_model.state_dict().items())

    if args.shared_base:
        shared_dir = Path(args.shared_base)
        base_ckpt_dir = shared_dir
        base_ckpt_path = shared_dir / "final_base.pt"
        init_path = shared_dir / "initial_state.pt"
        if not base_ckpt_path.exists():
            raise FileNotFoundError(f"Shared base final_base.pt missing at {base_ckpt_path}")
        base_data = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        # ONLY load history for methods that need it. Even then, use a lazy
        # iterator so we don't hold 35 GB in RAM at once (M4 unified memory
        # ~16-32 GB). Downstream baselines iterate the history exactly once.
        class _LazyHistory:
            def __init__(self, files):
                self.files = files
            def __iter__(self):
                for p in self.files:
                    yield torch.load(p, map_location="cpu", weights_only=False)
            def __len__(self):
                return len(self.files)
            def __bool__(self):
                return len(self.files) > 0
        history = []
        if args.method in HISTORY_METHODS:
            hist_dir = shared_dir / "history"
            legacy_hist = shared_dir / "history.pt"
            if hist_dir.exists():
                files = sorted(hist_dir.glob("r*.pt"), key=lambda q: int(q.stem[1:]))
                history = _LazyHistory(files)
            elif legacy_hist.exists():
                history = torch.load(legacy_hist, map_location="cpu", weights_only=False)
        if args.method == "federaser":
            if init_path.exists():
                initial_state = torch.load(init_path, map_location="cpu", weights_only=False)
            else:
                logger.warning("shared base has no initial_state.pt; using a fresh init (may drift from history's theta_0)")
        result = {"global_state": base_data["global_state"],
                  "round_index": base_data.get("round_index", cfg["federated"]["total_rounds"]),
                  "history": history,
                  "participation": base_data.get("participation", [])}
        logger.info(f"Reusing shared base at {shared_dir}. Skipping FedAvg training.")
    else:
        base_ckpt_dir = ckpt_root / "base"
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
            checkpoint_dir=base_ckpt_dir,
            seed=exp["seed"],
            store_history=(args.method in HISTORY_METHODS),
            evaluate_every=cfg["evaluation"]["evaluate_every"],
            evaluate_fn=evaluate,
            checkpoint_every=cfg.section("evaluation").get("checkpoint_every", 1),
            resume=True,
        )
        # train_fedavg_and_maybe_store_history now returns a history_dir
        # pointer (not an eagerly-loaded list). Load from disk here.
        hd = result.get("history_dir")
        if hd:
            hd_p = Path(hd)
            result["history"] = [torch.load(p, map_location="cpu", weights_only=False)
                                 for p in sorted(hd_p.glob("r*.pt"), key=lambda q: int(q.stem[1:]))]
        else:
            result["history"] = []

    logger.info("Base training complete/loaded. Applying baseline unlearning.")

    # ---- Apply baseline --------------------------------------------------
    forget_client = args.forget_client
    forget_ds = fcl_split.client_dataset(forget_client, task_id=None)

    tic = time.time()
    if args.method == "halimi":
        out = halimi_unlearn(
            global_state=result["global_state"],
            reference_state=None,
            model_factory=lambda: model_factory().to(exp["device"]),
            forget_dataset=forget_ds,
            device=exp["device"],
            ascent_steps=cfg.section("baselines").get("halimi_ascent_steps", 50),
            ascent_lr=cfg.section("baselines").get("halimi_ascent_lr", 1e-3),
            radius=cfg.section("baselines").get("halimi_radius", 3.0),
            batch_size=cfg["federated"]["batch_size"],
        )
    elif args.method == "fedmemprune":
        # Initial state: same-seed fresh model.
        init_model = model_factory().to(exp["device"])
        init_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in init_model.state_dict().items())
        out = fedmemprune_unlearn(
            global_state=result["global_state"],
            initial_state=init_state,
            model_factory=lambda: model_factory().to(exp["device"]),
            forget_dataset=forget_ds,
            device=exp["device"],
            prune_quantile=cfg.section("baselines").get("fedmemprune_quantile", 0.95),
            reset_scale=cfg.section("baselines").get("fedmemprune_reset_scale", 0.0),
            include_head=cfg.section("baselines").get("fedmemprune_include_head", True),
        )
    elif args.method == "fedrecovery":
        out = fedrecovery_unlearn(
            global_state=result["global_state"],
            history=result["history"],
            forget_client_id=forget_client,
            weighting=cfg.section("baselines").get("fedrecovery_weighting", "linear"),
            eta=cfg.section("baselines").get("fedrecovery_eta", 1.0),
            dp_eps=(args.fedrecovery_eps if args.fedrecovery_eps is not None
                    else cfg.section("baselines").get("fedrecovery_eps", 1.0)),
            dp_delta=cfg.section("baselines").get("fedrecovery_delta", 1e-5),
            dp_sensitivity=cfg["federated"]["clip_norm"],
        )
    elif args.method == "federaser":
        out = federaser_unlearn(
            initial_state=initial_state,
            history=result["history"],
            forget_client_id=forget_client,
            lr_calib=cfg.section("baselines").get("federaser_lr_calib", 0.5),
        )
    elif args.method == "sifu":
        # Reconstruct rounds_checkpoints from base_ckpt_dir's numbered files.
        rounds_checkpoints = {}
        for ck in sorted(base_ckpt_dir.glob("ckpt_r*.pt")):
            r = int(ck.stem.split("_r")[-1])
            data = torch.load(ck, map_location="cpu", weights_only=False)
            rounds_checkpoints[r] = data["global_state"]

        # Proper SIFU replay: fine-tune from the rollback checkpoint for the
        # remaining rounds using fresh local gradients on the retained clients,
        # not stored deltas. The previous implementation applied the client's
        # historical deltas (which were themselves computed on a model that
        # had seen the forget client at earlier rounds), introducing a bias
        # that collapsed utility to chance. The reviewer specifically asked us
        # to fix this (Q2 / W8).
        from src.federated.aggregator import DecayedAggregator, DecayState
        from src.federated.server import Server as _Server
        from src.federated.client import Client as _Client
        from src.federated.loop import _sample_clients
        from src.federated.params import add_state, zero_state_like
        import numpy as _np

        def resume_fn(state, from_r, to_r, exclude_cid):
            # Vanilla FedAvg replay: alpha≈0, no noise, excluding the target client.
            agg = DecayedAggregator(alpha=1e-6, server_lr=1.0, server_sigma=0.0, seed=exp["seed"])
            srv = _Server(
                global_state=OrderedDict((k, v.clone()) for k, v in state.items()),
                decay_state=DecayState.zeros_from(state, alpha=1e-6),
                aggregator=agg,
            )
            # Reconstruct client-sampling for rounds from_r+1..to_r. Reseed the
            # RNG and advance it through the trained rounds so sampling matches
            # the original substrate.
            rng = _np.random.default_rng(exp["seed"])
            # Fast-forward RNG to round from_r.
            for _ in range(from_r + 1):
                _sample_clients(0, fcl_split.num_clients, cfg["federated"]["client_sample_frac"], rng)
            text_mode = (fcl_split.modality == "text")
            for r in range(from_r + 1, to_r + 1):
                sampled = _sample_clients(r, fcl_split.num_clients, cfg["federated"]["client_sample_frac"], rng)
                sampled = [c for c in sampled if c != exclude_cid]
                if not sampled:
                    continue
                task_id = min(r // cfg["federated"]["rounds_per_task"], fcl_split.num_tasks - 1)
                updates = []
                from src.federated.client import ClientUpdate as _Update
                for cid in sampled:
                    ds = fcl_split.client_dataset(cid, task_id=task_id)
                    if len(ds) == 0:
                        continue
                    c = _Client(
                        client_id=cid, dataset=ds,
                        local_epochs=cfg["federated"]["local_epochs"],
                        batch_size=cfg["federated"]["batch_size"],
                        lr=cfg["federated"]["lr"],
                        clip_norm=cfg["federated"]["clip_norm"],
                        device=exp["device"], text_mode=text_mode,
                    )
                    m = model_factory().to(exp["device"])
                    updates.append(c.local_update(srv.global_state, m, round_index=r))
                srv.apply_round(updates)
            return srv.global_state

        out = sifu_unlearn(
            rounds_checkpoints=rounds_checkpoints,
            total_rounds=cfg["federated"]["total_rounds"],
            forget_client_id=forget_client,
            resume_train_fn=resume_fn,
            target_eps=cfg.section("baselines").get("sifu_target_eps", 1.0),
            clip_C=cfg["federated"]["clip_norm"],
            lr=cfg["federated"]["lr"],
            participation_rho=cfg.section("baselines").get("participation_rho", 0.3),
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"{args.method} unlearning: wall_time={out.wall_time_s:.3f}s, extras={out.extras}")

    out_path = Path(args.output) if args.output else ckpt_root / f"final_unlearned_c{forget_client}.pt"
    torch.save({
        "global_state": out.unlearned_state,
        "method": out.method,
        "guarantee": out.guarantee,
        "extras": out.extras,
        "wall_time_s": out.wall_time_s,
        "base_global_state": result["global_state"],  # so eval can compare pre vs post
    }, out_path)
    # Also save the base ("pre-unlearning") checkpoint for the eval step.
    base_final_path = ckpt_root / "base_final.pt"
    torch.save({"global_state": result["global_state"],
                "history_size_bytes": sum(sys.getsizeof(entry["deltas"]) for entry in result["history"])},
               base_final_path)
    logger.info(f"Saved: {out_path} (post-unlearn) and {base_final_path} (pre-unlearn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
