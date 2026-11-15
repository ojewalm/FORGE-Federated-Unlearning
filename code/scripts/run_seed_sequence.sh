#!/usr/bin/env bash
# Chain seed 1 and seed 2 runs after seed 0 completes.
# Waits for runs/cifar10_real/benchmark_c3.csv, then launches seed 1,
# waits for seed 1, then launches seed 2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[chain] waiting for seed-0 benchmark CSV..."
until [ -f runs/cifar10_real/benchmark_c3.csv ]; do sleep 60; done
echo "[chain] seed-0 CSV found, starting seed 1..."

python3 scripts/benchmark.py --config configs/cifar10_s1.yaml --forget-client 3 \
    --skip-existing --run-lira --lira-n-shadow 8 --run-fuia \
    --fuia-n-samples 4 --fuia-steps 200 2>&1 | tee runs/seed1.log

echo "[chain] seed 1 CSV found, starting seed 2..."
python3 scripts/benchmark.py --config configs/cifar10_s2.yaml --forget-client 3 \
    --skip-existing --run-lira --lira-n-shadow 8 --run-fuia \
    --fuia-n-samples 4 --fuia-steps 200 2>&1 | tee runs/seed2.log

echo "[chain] all three seeds complete. Aggregating..."
python3 scripts/aggregate_results.py \
    runs/cifar10_real/benchmark_c3.json \
    runs/cifar10_s1/benchmark_c3.json \
    runs/cifar10_s2/benchmark_c3.json \
    --out runs/aggregated_3seed.csv
echo "[chain] done. Aggregated: runs/aggregated_3seed.csv"
