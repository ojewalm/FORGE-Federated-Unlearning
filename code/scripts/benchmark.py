"""Run the full benchmark: five baselines + our combined method + retrain gold.

Produces a CSV comparison table indexed by method. Each row includes:
  - avg test accuracy (utility)
  - MIA advantage (Yeom) on the departed client's data (forgetting)
  - Retrain-KL vs retrain-from-scratch gold (behavioural distance)
  - Wall-clock unlearning latency
  - Formal guarantee (eps, delta) if reported
  - Storage requirement (bytes retained) if the method needs history

Usage:
    python scripts/benchmark.py --config configs/cifar100_smoke.yaml --forget-client 3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Subset

from src.attacks import mia_yeom, mia_lira, mia_lira_score, prepare_lira_shadows, fuia_attack
from src.datasets import build_dataset
from src.metrics import average_test_accuracy
from src.metrics.forgetting import retrain_kl
from src.models import build_model
from src.utils import load_config, resolve_device, set_seed, setup_logger


ALL_METHODS = [
    "retrain", "no_unlearn", "halimi", "fedmemprune",
    "fedrecovery", "fedrecovery_eps2", "fedrecovery_eps5", "fedrecovery_eps10",
    "fedrecovery_eps_inf",   # no-noise anchor (reviewer W4/W7)
    "federaser", "sifu", "ours",
]


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


def _run(cmd: list[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _score(
    m_post: torch.nn.Module, m_retrain: torch.nn.Module | None,
    fcl_split, forget_client: int, seed: int, device: str,
    *, shadow_bundle: dict | None = None,
) -> dict:
    """Common eval: accuracy, Yeom MIA, retrain-KL, optional LiRA via a
    pre-prepared shadow bundle (so we don't retrain shadows per method)."""
    acc = average_test_accuracy(m_post, fcl_split, device=device)
    client_idx = fcl_split.client_indices[forget_client].tolist()
    member = Subset(fcl_split.train_set, client_idx)
    others = np.concatenate(
        [fcl_split.client_indices[i] for i in range(fcl_split.num_clients) if i != forget_client]).tolist()
    rng = np.random.default_rng(seed)
    non_idx = rng.choice(others, size=min(len(client_idx), len(others)), replace=False).tolist()
    non = Subset(fcl_split.train_set, non_idx)
    mia_y = mia_yeom(m_post, member, non, device=device)
    out = {"acc": round(acc, 4), "mia_yeom_adv": round(mia_y["advantage"], 4)}
    if m_retrain is not None:
        probe = Subset(fcl_split.test_set, list(range(min(256, len(fcl_split.test_set)))))
        out["kl_vs_retrain"] = round(retrain_kl(m_post, m_retrain, probe, device=device), 4)
    else:
        out["kl_vs_retrain"] = None
    if shadow_bundle is not None:
        lira = mia_lira_score(target_model=m_post, shadow_bundle=shadow_bundle, device=device)
        out["mia_lira_auroc"] = round(lira["auroc"], 4)
        out["mia_lira_advantage"] = round(lira["advantage"], 4)
        out["mia_lira_tpr_at_1pct"] = round(lira["tpr_at_fpr_0_01"], 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget-client", type=int, required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS,
                    help=f"Subset of {ALL_METHODS}. Default: all.")
    ap.add_argument("--r-forget", type=int, default=None,
                    help="For 'ours', round at which client departed. Defaults to total_rounds//3.")
    ap.add_argument("--r-request", type=int, default=None,
                    help="For 'ours', round at which erasure requested. Defaults to total_rounds.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a method if its output artefact already exists.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-lira", action="store_true",
                    help="Enable LiRA shadow-model MIA (slow; adds n_shadow trainings per row).")
    ap.add_argument("--lira-n-shadow", type=int, default=8,
                    help="Number of shadow models for LiRA (default 8).")
    ap.add_argument("--run-fuia", action="store_true",
                    help="Enable FUIA reconstruction attack against ours/retrain/fedrecovery (slow).")
    ap.add_argument("--fuia-n-samples", type=int, default=8,
                    help="Number of forget-set samples to attempt FUIA reconstruction on.")
    ap.add_argument("--fuia-steps", type=int, default=500,
                    help="Optimisation steps per FUIA reconstruction.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    exp["device"] = resolve_device(exp.get("device", "auto"))
    out_root = Path(args.output_dir or cfg["output"]["checkpoint_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("fu-bench", log_dir=out_root)

    total_rounds = cfg["federated"]["total_rounds"]
    r_forget = args.r_forget or total_rounds // 3
    r_request = args.r_request or total_rounds
    fc = args.forget_client
    root = out_root

    # ---- Ensure every artefact exists ------------------------------------
    # 1. Retrain-from-scratch.
    retrain_final = root / f"retrain_excl_c{fc}" / "final.pt"
    if "retrain" in args.methods and (not args.skip_existing or not retrain_final.exists()):
        _run([sys.executable, "scripts/retrain.py", "--config", args.config, "--exclude-client", str(fc)])

    # 2. Our combined method: train + unlearn.
    our_final = root / "final.pt"
    our_unlearned = root / f"final_unlearned_c{fc}.pt"
    if "ours" in args.methods:
        if not args.skip_existing or not our_final.exists():
            _run([sys.executable, "scripts/train.py", "--config", args.config])
        if not args.skip_existing or not our_unlearned.exists():
            _run([sys.executable, "scripts/unlearn.py", "--config", args.config,
                  "--checkpoint", str(our_final), "--client", str(fc),
                  "--r-forget", str(r_forget), "--r-request", str(r_request),
                  "--output", str(our_unlearned)])

    # 3. Baselines. Train a single shared FedAvg-with-history base and reuse
    # it across all five baselines. This makes the comparison apples-to-apples
    # (same base, different unlearning steps) and cuts wall time ~5x.
    # FedRecovery variants cover the eps-sweep requested by reviewers.
    baseline_map: dict[str, dict] = {
        "halimi":            {"path": root / f"baseline_halimi_c{fc}" / f"final_unlearned_c{fc}.pt",            "method": "halimi",      "eps": None},
        "fedmemprune":       {"path": root / f"baseline_fedmemprune_c{fc}" / f"final_unlearned_c{fc}.pt",       "method": "fedmemprune", "eps": None},
        "fedrecovery":       {"path": root / f"baseline_fedrecovery_c{fc}" / f"final_unlearned_c{fc}.pt",       "method": "fedrecovery", "eps": None},
        "fedrecovery_eps2":  {"path": root / f"baseline_fedrecovery_c{fc}_eps2.0" / f"final_unlearned_c{fc}.pt","method": "fedrecovery", "eps": 2.0},
        "fedrecovery_eps5":  {"path": root / f"baseline_fedrecovery_c{fc}_eps5.0" / f"final_unlearned_c{fc}.pt","method": "fedrecovery", "eps": 5.0},
        "fedrecovery_eps10": {"path": root / f"baseline_fedrecovery_c{fc}_eps10.0" / f"final_unlearned_c{fc}.pt","method": "fedrecovery","eps": 10.0},
        "fedrecovery_eps_inf": {"path": root / f"baseline_fedrecovery_c{fc}_epsinf" / f"final_unlearned_c{fc}.pt","method": "fedrecovery","eps": float("inf")},
        "federaser":         {"path": root / f"baseline_federaser_c{fc}" / f"final_unlearned_c{fc}.pt",         "method": "federaser",   "eps": None},
        "sifu":              {"path": root / f"baseline_sifu_c{fc}" / f"final_unlearned_c{fc}.pt",              "method": "sifu",        "eps": None},
    }
    baseline_methods_requested = [m for m in baseline_map if m in args.methods]
    if baseline_methods_requested:
        shared_base_dir = root / "shared_base"
        shared_final = shared_base_dir / "final_base.pt"
        if not args.skip_existing or not shared_final.exists():
            _run([sys.executable, "scripts/train_shared_base.py", "--config", args.config,
                  "--output", str(shared_base_dir)])
        for method_tag in baseline_methods_requested:
            info = baseline_map[method_tag]
            if args.skip_existing and info["path"].exists():
                continue
            cmd = [sys.executable, "scripts/run_baseline.py", "--config", args.config,
                   "--method", info["method"], "--forget-client", str(fc),
                   "--shared-base", str(shared_base_dir)]
            if info["eps"] is not None:
                cmd += ["--fedrecovery-eps", str(info["eps"])]
            _run(cmd)

    # ---- Evaluate all methods --------------------------------------------
    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_kwargs = _model_kwargs(cfg, fcl_split.num_classes)

    def load_model(state):
        m = build_model(cfg["model"]["name"], **model_kwargs).to(exp["device"])
        m.load_state_dict(state)
        return m

    m_retrain = None
    if retrain_final.exists():
        m_retrain = load_model(_load_state(retrain_final)["global_state"])

    rows: list[dict] = []

    # LiRA: train shadow fleet once; score each target model against it.
    shadow_bundle = None
    if args.run_lira:
        def _lira_factory():
            return build_model(cfg["model"]["name"], **model_kwargs)
        client_idx_lira = fcl_split.client_indices[fc].tolist()
        member_lira = Subset(fcl_split.train_set, client_idx_lira)
        others = np.concatenate(
            [fcl_split.client_indices[i] for i in range(fcl_split.num_clients) if i != fc]).tolist()
        rng_lira = np.random.default_rng(exp["seed"])
        non_idx = rng_lira.choice(others, size=min(len(client_idx_lira), len(others)), replace=False).tolist()
        non_lira = Subset(fcl_split.train_set, non_idx)
        logger.info(f"[lira] training {args.lira_n_shadow} shadow models (once, shared across all methods)")
        shadow_bundle = prepare_lira_shadows(
            pool_dataset=fcl_split.train_set,
            query_members=member_lira,
            query_nonmembers=non_lira,
            model_factory=_lira_factory,
            device=exp["device"],
            n_shadow=args.lira_n_shadow,
            shadow_epochs=5,
            seed=exp["seed"],
        )

    def _score_full(m_post, m_retrain):
        return _score(m_post, m_retrain, fcl_split, fc, exp["seed"], exp["device"],
                      shadow_bundle=shadow_bundle)

    # retrain gold (self-vs-self kl = 0).
    if "retrain" in args.methods and m_retrain is not None:
        rows.append({"method": "retrain", **_score_full(m_retrain, m_retrain),
                     "wall_time_s": None, "guarantee": "exact", "storage": "n/a"})

    # no_unlearn: the pre-unlearn model.
    no_unlearn_path = our_final if our_final.exists() else None
    if not no_unlearn_path:
        for method in ("halimi", "fedmemprune"):
            p = root / f"baseline_{method}_c{fc}" / "base_final.pt"
            if p.exists():
                no_unlearn_path = p
                break
    if "no_unlearn" in args.methods and no_unlearn_path is not None:
        m_no = load_model(_load_state(no_unlearn_path)["global_state"])
        rows.append({"method": "no_unlearn", **_score_full(m_no, m_retrain),
                     "wall_time_s": 0.0, "guarantee": "none", "storage": "n/a"})

    # Ours.
    if "ours" in args.methods and our_unlearned.exists():
        data = _load_state(our_unlearned)
        m_o = load_model(data["global_state"])
        rows.append({"method": "ours", **_score_full(m_o, m_retrain),
                     "wall_time_s": data.get("certificate", {}).get("wall_time_s"),
                     "guarantee": f"eps={data.get('certificate',{}).get('eps_total','?')}",
                     "storage": "O(1) per client (summary only)"})

    # Baselines (including FedRecovery eps-sweep variants).
    for method_tag, info in baseline_map.items():
        if method_tag not in args.methods or not info["path"].exists():
            continue
        data = _load_state(info["path"])
        m_b = load_model(data["global_state"])
        guarantee = (f"eps={data['guarantee'].get('eps')}" if data.get("guarantee") else "none")
        storage = ("O(N*R*|θ|) history" if data.get("extras", {}).get("requires_history_storage")
                   else ("O(R*|θ|) checkpoints" if data.get("extras", {}).get("requires_checkpoint_storage")
                         else "O(1)"))
        rows.append({"method": method_tag, **_score_full(m_b, m_retrain),
                     "wall_time_s": data.get("wall_time_s"),
                     "guarantee": guarantee, "storage": storage})

    # Optional: FUIA. Run only for ours, fedrecovery, retrain (and no_unlearn
    # as a positive control). FUIA is slow (minutes per sample); gated.
    if args.run_fuia:
        from torch.utils.data import Subset as _Subset
        forget_set = _Subset(fcl_split.train_set,
                             fcl_split.client_indices[fc].tolist()[:args.fuia_n_samples])
        input_shape = fcl_split.input_shape
        fuia_targets = []
        if "retrain" in args.methods and m_retrain is not None:
            fuia_targets.append(("retrain", m_retrain, m_retrain))   # pair == same model
        if "no_unlearn" in args.methods and no_unlearn_path is not None:
            m_no = load_model(_load_state(no_unlearn_path)["global_state"])
            fuia_targets.append(("no_unlearn", m_no, m_no))
        if "ours" in args.methods and our_unlearned.exists() and our_final.exists():
            m_pre = load_model(_load_state(our_final)["global_state"])
            m_post = load_model(_load_state(our_unlearned)["global_state"])
            fuia_targets.append(("ours", m_pre, m_post))
        for tag in ("fedrecovery", "fedrecovery_eps10"):
            if tag in args.methods and baseline_map[tag]["path"].exists():
                data = _load_state(baseline_map[tag]["path"])
                m_pre = load_model(data["base_global_state"]) if "base_global_state" in data else None
                m_post = load_model(data["global_state"])
                if m_pre is not None:
                    fuia_targets.append((tag, m_pre, m_post))
        fuia_results: dict = {}
        for tag, m_pre, m_post in fuia_targets:
            logger.info(f"[fuia] running against {tag} (n_samples={args.fuia_n_samples})")
            res = fuia_attack(model_pre=m_pre, model_post=m_post,
                              forget_dataset=forget_set, num_classes=fcl_split.num_classes,
                              input_shape=input_shape, device=exp["device"],
                              n_samples=args.fuia_n_samples, steps=args.fuia_steps)
            fuia_results[tag] = res
            # Attach to the matching row.
            for row in rows:
                if row["method"] == tag:
                    row["fuia_psnr"] = round(res["mean_psnr"], 2)
                    row["fuia_cos"]  = round(res["mean_cosine_sim"], 4)
        (out_root / f"fuia_c{fc}.json").write_text(json.dumps(fuia_results, indent=2, default=str))

    # ---- Write CSV + JSON summary ---------------------------------------
    out_csv = out_root / f"benchmark_c{fc}.csv"
    all_keys = sorted({k for row in rows for k in row.keys()})
    # Canonical column ordering.
    priority = ["method","acc","mia_yeom_adv","mia_lira_auroc","mia_lira_advantage",
                "mia_lira_tpr_at_1pct","kl_vs_retrain","fuia_psnr","fuia_cos",
                "wall_time_s","guarantee","storage"]
    ordered = [k for k in priority if k in all_keys] + [k for k in all_keys if k not in priority]
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ordered})
    out_json = out_root / f"benchmark_c{fc}.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))
    print("\n=== Benchmark results ===")
    short_hdr = ["method","acc","mia_yeom_adv","mia_lira_advantage","kl_vs_retrain",
                 "fuia_psnr","wall_time_s","guarantee","storage"]
    print(" | ".join(f"{h:>18}" for h in short_hdr))
    print("-" * (19 * len(short_hdr)))
    for row in rows:
        print(" | ".join(f"{str(row.get(h, ''))[:18]:>18}" for h in short_hdr))
    print(f"\nWrote: {out_csv}\n       {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
