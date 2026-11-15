"""Measure backdoor attack success rate (ASR) before and after FORGE erasure.

Expects the training config to contain a `backdoor` section (see
`src/federated/backdoor.py`). Loads the pre-erasure and post-erasure heads from
the standard training/unlearn output locations and reports ASR for each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.federated.backdoor import BackdoorConfig, attack_success_rate
from src.models import build_model
from src.utils import load_config, resolve_device, set_seed


def _dataset_kwargs(cfg) -> dict:
    ds = cfg.section("dataset").to_dict()
    ds.pop("name", None)
    ds["data_root"] = cfg["experiment"]["data_root"]
    ds["seed"] = cfg["experiment"]["seed"]
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget-client", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg.section("experiment").to_dict()
    set_seed(exp["seed"])
    device = resolve_device(exp.get("device", "auto"))
    exp["device"] = device

    fcl_split = build_dataset(cfg["dataset"]["name"], **_dataset_kwargs(cfg))
    model_factory = lambda: build_model(
        name=cfg["model"]["name"],
        num_classes=fcl_split.num_classes,
        in_channels=cfg["model"].get("in_channels", 3),
        small_input=cfg["model"].get("small_input", True),
    )

    bd_cfg = cfg.section("backdoor")
    bd = BackdoorConfig(
        client_id=int(bd_cfg.get("client_id", args.forget_client)),
        rounds_before_erasure=int(bd_cfg.get("rounds_before_erasure", 20)),
        target_label=int(bd_cfg.get("target_label", 0)),
        patch_size=int(bd_cfg.get("patch_size", 4)),
        patch_value=float(bd_cfg.get("patch_value", 2.5)),
    )

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    pre = ckpt_dir / "final.pt"
    post = ckpt_dir / f"final_unlearned_c{args.forget_client}.pt"

    def load_model(path: Path):
        model = model_factory().to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        global_state = state.get("global_state", state)
        model.load_state_dict(global_state)
        return model

    test_loader = DataLoader(fcl_split.test_set, batch_size=128, shuffle=False)

    results = {}
    for tag, path in [("pre_erasure", pre), ("post_erasure", post)]:
        if not path.exists():
            print(f"[eval_backdoor] missing {path}; skipping {tag}")
            continue
        model = load_model(path)
        asr = attack_success_rate(model, test_loader, bd, device)
        results[tag] = asr
        print(f"[eval_backdoor] {tag}: ASR = {asr:.4f}")

    if "pre_erasure" in results and "post_erasure" in results:
        results["asr_reduction"] = results["pre_erasure"] - results["post_erasure"]
        print(f"[eval_backdoor] ASR reduction (pre - post): {results['asr_reduction']:.4f}")

    out_path = Path(args.output) if args.output else ckpt_dir / f"backdoor_c{args.forget_client}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_backdoor] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
