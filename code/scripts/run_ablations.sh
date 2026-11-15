#!/usr/bin/env bash
# Run the ablation sweep (alpha, sigma, summary type) requested by reviewers.
# Each ablation reuses the shared retrain and shared base from the main run.
# Produces one benchmark CSV per ablation cell under runs/cifar10_abl_<cell>.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_CFG="configs/cifar10_real.yaml"

# ---- Alpha sweep ----
for alpha in 0.1 0.5 0.9; do
  OUT="runs/cifar10_abl_alpha${alpha}"
  CFG="configs/cifar10_abl_alpha${alpha}.yaml"
  python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['aggregator']['alpha'] = $alpha
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_abl_alpha_$alpha'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
  python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
      --methods ours retrain no_unlearn
done

# ---- Summary-type sweep (moments vs logits) ----
for summ in moments logits; do
  OUT="runs/cifar10_abl_summary_${summ}"
  CFG="configs/cifar10_abl_summary_${summ}.yaml"
  python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['summary']['type'] = '$summ'
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_abl_summary_$summ'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
  python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
      --methods ours retrain no_unlearn
done

# ---- Sigma sweep ----
for sigma in 0.0001 0.001 0.01; do
  OUT="runs/cifar10_abl_sigma${sigma}"
  CFG="configs/cifar10_abl_sigma${sigma}.yaml"
  python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['aggregator']['server_sigma'] = $sigma
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_abl_sigma_$sigma'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
  python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
      --methods ours retrain no_unlearn
done

echo "Ablation sweep complete."
