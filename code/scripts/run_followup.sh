#!/usr/bin/env bash
# After FEMNIST seed 0 completes, run:
#   (a) LiRA n_shadow=32 rerun of CIFAR-10 seed 0 (reviewer suggestion 5)
#   (b) Full-budget FUIA of FORGE and FedRecovery eps=inf (reviewer suggestion 6)
# then re-aggregate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[followup] waiting for FEMNIST seed-0 CSV..."
until [ -f runs/femnist_real/benchmark_c3.csv ]; do sleep 60; done
echo "[followup] FEMNIST CSV found."

# (a) LiRA-32 rerun of CIFAR-10 seed 0.
#   Blow away the previous CIFAR-10 benchmark CSV so it rebuilds with the
#   stronger shadow pool. All training artefacts are preserved via
#   --skip-existing.
echo "[followup] starting CIFAR-10 LiRA-32 rerun (seed 0)..."
rm -f runs/cifar10_real/benchmark_c3.csv runs/cifar10_real/benchmark_c3.json runs/cifar10_real/fuia_c3.json
python3 scripts/benchmark.py --config configs/cifar10_real.yaml --forget-client 3 \
    --skip-existing --run-lira --lira-n-shadow 32 --run-fuia \
    --fuia-n-samples 16 --fuia-steps 1000 2>&1 | tee runs/cifar10_lira32.log
echo "[followup] LiRA-32 rerun done."

# (b) Separate full-budget FUIA on the fresh benchmark is already included
# in the run above (--fuia-n-samples 16 --fuia-steps 1000).

# (c) Re-aggregate: seed 0 strong + seeds 1, 2 original.
echo "[followup] re-aggregating..."
python3 scripts/aggregate_results.py \
    runs/cifar10_real/benchmark_c3.json \
    runs/cifar10_s1/benchmark_c3.json \
    runs/cifar10_s2/benchmark_c3.json \
    --out runs/aggregated_3seed_v3.csv

echo "[followup] done."
echo "  CIFAR-10 LiRA-32 CSV: runs/cifar10_real/benchmark_c3.csv"
echo "  FEMNIST seed-0 CSV:   runs/femnist_real/benchmark_c3.csv"
echo "  Aggregated:           runs/aggregated_3seed_v3.csv"
