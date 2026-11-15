"""Print a status summary of an in-progress or completed benchmark run.

Usage:
    python scripts/status.py runs/cifar100_real
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# Need src on the path so pickle can find ResumableSnapshot.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.federated.checkpointing  # noqa: F401  needed for pickle deserialisation


def _fmt_ago(ts: float) -> str:
    dt = time.time() - ts
    if dt < 60:
        return f"{int(dt)}s ago"
    if dt < 3600:
        return f"{int(dt/60)}m ago"
    return f"{dt/3600:.1f}h ago"


def _describe_snapshot(path: Path) -> str:
    try:
        with path.open("rb") as f:
            s = pickle.load(f)
        r = getattr(s, "round_index", None) or getattr(s, "get", lambda k, d=None: d)("round_index", None)
        saved = getattr(s, "saved_at", path.stat().st_mtime)
        n_summ = len(getattr(s, "summaries", {}))
        return f"round={r} summaries={n_summ} saved={_fmt_ago(saved)}"
    except Exception as e:
        return f"(unreadable: {e})"


def _describe_dir(d: Path) -> str:
    if not d.exists():
        return "not started"
    # Prefer final*.pt over resume.pt: if a final artefact exists AND is newer
    # than the resume snapshot, the phase is complete. Otherwise resume.pt
    # indicates an in-progress or interrupted run.
    finals = list(d.glob("final*.pt"))
    resume = d / "resume.pt"
    if finals:
        latest_final = max(finals, key=lambda p: p.stat().st_mtime)
        if not resume.exists() or latest_final.stat().st_mtime >= resume.stat().st_mtime:
            return f"complete: {latest_final.name} saved {_fmt_ago(latest_final.stat().st_mtime)}"
    if resume.exists():
        return f"in-progress: {_describe_snapshot(resume)}"
    ckpts = sorted(d.glob("ckpt_r*.pt"))
    if ckpts:
        latest = ckpts[-1]
        return f"partial: {latest.name} saved {_fmt_ago(latest.stat().st_mtime)}"
    return "started, no artefact yet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--forget-client", type=int, default=3)
    args = ap.parse_args()

    d = args.run_dir
    fc = args.forget_client
    if not d.exists():
        print(f"Run dir does not exist: {d}")
        return 1

    print(f"=== {d} (forget client {fc}) ===\n")

    phases = [
        ("ours (train)",        d),
        ("ours (unlearn)",      d / f"final_unlearned_c{fc}.pt"),
        ("retrain-excl",        d / f"retrain_excl_c{fc}"),
        ("shared base",         d / "shared_base"),
        ("halimi",              d / f"baseline_halimi_c{fc}"),
        ("fedmemprune",         d / f"baseline_fedmemprune_c{fc}"),
        ("fedrecovery",         d / f"baseline_fedrecovery_c{fc}"),
        ("federaser",           d / f"baseline_federaser_c{fc}"),
        ("sifu",                d / f"baseline_sifu_c{fc}"),
    ]
    for label, p in phases:
        if p.is_file():
            if p.exists():
                print(f"  {label:<20} complete ({p.name}, {_fmt_ago(p.stat().st_mtime)})")
            else:
                print(f"  {label:<20} not started")
        else:
            print(f"  {label:<20} {_describe_dir(p)}")

    bench_csv = d / f"benchmark_c{fc}.csv"
    if bench_csv.exists():
        print(f"\nBenchmark CSV: {bench_csv} ({_fmt_ago(bench_csv.stat().st_mtime)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
