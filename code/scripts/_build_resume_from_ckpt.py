"""One-shot helper: synthesise a resume.pt from the highest ckpt_rN.pt.

Reads the newest ckpt_rN.pt in a checkpoint directory, captures a fresh RNG
state, and writes a ResumableSnapshot to resume.pt so that federated_train's
existing resume logic will pick up at round N on the next launch.

Trade-off: the restored RNG state is fresh (seeded deterministically from the
experiment seed + round), not the exact state at the moment of the crash. The
resulting continuation is statistically equivalent, not bit-identical.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.federated.checkpointing import ResumableSnapshot, RngState, save_snapshot


def _highest_ckpt(dir_: Path) -> tuple[Path, int]:
    best = None
    best_r = -1
    for p in dir_.glob("ckpt_r*.pt"):
        m = re.match(r"ckpt_r(\d+)\.pt$", p.name)
        if not m:
            continue
        r = int(m.group(1))
        if r > best_r:
            best_r = r
            best = p
    if best is None:
        raise FileNotFoundError(f"no ckpt_rN.pt in {dir_}")
    return best, best_r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="Experiment seed to re-seed the fresh RNG with.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    src, round_idx = _highest_ckpt(ckpt_dir)
    ck = torch.load(src, map_location="cpu", weights_only=False)

    # Re-seed deterministically from (seed, round_idx) so a re-run after a crash
    # is reproducible, even if not bit-identical to a non-crashed run.
    salt = args.seed * 1_000_000 + round_idx
    random.seed(salt)
    np.random.seed(salt)
    torch.manual_seed(salt)

    rng = RngState.capture()
    # Fresh aggregator generator (DecayedAggregator seeds its own Generator at
    # init; we hand it a fresh snapshot so re-seeding at round N is clean).
    gen = torch.Generator()
    gen.manual_seed(salt + 1)
    gen_state = gen.get_state()

    snap = ResumableSnapshot(
        round_index=round_idx,
        global_state=ck["global_state"],
        decay_tilde_delta=ck["decay_state"],
        decay_alpha=float(ck["decay_alpha"]),
        aggregator_gen_state=gen_state,
        summaries=ck.get("summaries", {}),
        rng_state=rng,
        extras={"synthesized_from": src.name, "re_seed_salt": salt},
    )
    out = ckpt_dir / "resume.pt"
    save_snapshot(snap, out)
    print(f"wrote {out} (round={round_idx}, source={src.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
