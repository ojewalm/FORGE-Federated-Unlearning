"""Aggregate multiple benchmark_c*.json files into mean ± 95% CI per method.

Usage:
    python scripts/aggregate_results.py runs/cifar10_s*/benchmark_c*.json \
        --out runs/aggregated.csv

Produces one row per method with mean, std, and 95% CI across all (seed,
client) pairs for every numeric metric reported.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path


def _collect(paths: list[str]) -> dict[str, list[dict]]:
    """Group per-method rows across all files."""
    by_method: dict[str, list[dict]] = {}
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for row in data:
            m = row["method"]
            by_method.setdefault(m, []).append(row)
    return by_method


def _ci95(values: list[float]) -> tuple[float, float]:
    """Mean and 95% CI half-width via Student's-t at small n.

    For n <= 30 the normal approximation is systematically narrow. Use the
    Student's-t quantile with n-1 degrees of freedom (two-sided alpha=0.05).
    """
    # Two-sided t_{0.975, df} lookup (precomputed for small n).
    T_CRIT_95 = {
        1: float("inf"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
        6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.145, 20: 2.093, 30: 2.042,
    }
    if len(values) < 2:
        return (values[0], 0.0) if values else (float("nan"), 0.0)
    mu = statistics.fmean(values)
    sd = statistics.stdev(values)
    df = len(values) - 1
    # Pick nearest tabulated value, biasing conservative (use smaller df if between).
    tcrit = T_CRIT_95.get(df, 1.96 if df > 30 else T_CRIT_95[min([k for k in T_CRIT_95 if k >= df] or [1])])
    hw = tcrit * sd / math.sqrt(len(values))
    return mu, hw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Globs or file paths to benchmark_*.json files.")
    ap.add_argument("--out", default="runs/aggregated.csv")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p)) if "*" in p else [p])
    files = [f for f in files if Path(f).exists()]
    if not files:
        print("No input files matched.", file=sys.stderr)
        return 1

    print(f"Aggregating {len(files)} benchmark files:")
    for f in files:
        print(f"  {f}")

    rows_by_method = _collect(files)

    # Metrics to aggregate (numeric only).
    metric_keys = ["acc", "mia_yeom_adv", "mia_lira_auroc", "mia_lira_advantage",
                   "mia_lira_tpr_at_1pct", "kl_vs_retrain", "fuia_psnr", "fuia_cos",
                   "wall_time_s"]

    out_rows = []
    for method, rows in sorted(rows_by_method.items()):
        row_out = {"method": method, "n_runs": len(rows)}
        for k in metric_keys:
            values = [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")
                      and isinstance(r.get(k), (int, float))]
            if not values:
                continue
            mu, hw = _ci95(values)
            row_out[f"{k}_mean"] = round(mu, 4)
            row_out[f"{k}_ci95"] = round(hw, 4)
        out_rows.append(row_out)

    all_keys = sorted({k for r in out_rows for k in r.keys()})
    priority = ["method", "n_runs"]
    ordered = priority + [k for k in all_keys if k not in priority]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in ordered})
    print(f"Wrote aggregated results to {out_path}")

    # Pretty-print.
    print()
    print("=== Aggregated benchmark (mean ± 95% CI across runs) ===")
    hdr = ["method", "n", "acc", "mia_yeom", "mia_lira_auroc", "kl_vs_retrain"]
    print(" | ".join(f"{h:>15}" for h in hdr))
    for row in out_rows:
        acc = f"{row.get('acc_mean', '-'):.3f}±{row.get('acc_ci95', 0):.3f}"
        yeom = f"{row.get('mia_yeom_adv_mean', '-'):.3f}±{row.get('mia_yeom_adv_ci95', 0):.3f}"
        auroc = f"{row.get('mia_lira_auroc_mean', '-'):.3f}±{row.get('mia_lira_auroc_ci95', 0):.3f}" if "mia_lira_auroc_mean" in row else "-"
        kl = f"{row.get('kl_vs_retrain_mean', '-'):.3f}±{row.get('kl_vs_retrain_ci95', 0):.3f}" if "kl_vs_retrain_mean" in row else "-"
        print(f"{row['method']:>15} | {row['n_runs']:>15} | {acc:>15} | {yeom:>15} | {auroc:>15} | {kl:>15}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
