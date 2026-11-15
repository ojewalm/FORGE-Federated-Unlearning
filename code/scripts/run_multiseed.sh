#!/usr/bin/env bash
# Multi-seed, multi-client replication of the main benchmark.
# For each (seed, client) pair produces a benchmark_c{client}.csv.
# Aggregation into mean ± 95% CI is done by scripts/aggregate_results.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_CFG="configs/cifar10_real.yaml"
SEEDS=(0 1 2)
CLIENTS=(3 17 42)

for seed in "${SEEDS[@]}"; do
  for client in "${CLIENTS[@]}"; do
    OUT="runs/cifar10_s${seed}"
    CFG="configs/cifar10_s${seed}.yaml"
    python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['experiment']['seed'] = $seed
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_s$seed'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
    # Each seed shares its own retrain + ours + shared base; each client within
    # a seed just re-runs the baseline unlearn step and eval.
    python3 scripts/benchmark.py --config "$CFG" --forget-client "$client" --skip-existing \
        --run-lira --lira-n-shadow 4 || echo "warning: seed=$seed client=$client failed"
  done
done

echo "Multi-seed replication complete."
echo "Aggregate with: python3 scripts/aggregate_results.py runs/cifar10_s*/benchmark_c*.json"
