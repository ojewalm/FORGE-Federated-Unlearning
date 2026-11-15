#!/usr/bin/env bash
# Continuation of run_all_remaining.sh.
# Blocks until Phase 4's marker (aggregated_3seed_v4.csv) appears, then
# runs Phases 5-9:
#   Phase 5: W11 within-aggregator (FORGE at alpha=0, sigma=0).
#   Phase 6: W12 near-IID CIFAR-10 (dirichlet_alpha=10).
#   Phase 7: W6 CIFAR-100 full benchmark.
#   Phase 8: W6 Tiny-ImageNet full benchmark.
#   Phase 9: W6 20 Newsgroups full benchmark.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="runs/_remaining"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

MARKER="runs/aggregated_3seed_v4.csv"
MAIN_RETRAIN="runs/cifar10_real/retrain_excl_c3"
BASE_CFG="configs/cifar10_real.yaml"

log "=== run_queued_W6_W11_W12.sh starting; waiting for Phase 4 marker ==="
log "Waiting for: $MARKER"
while [ ! -f "$MARKER" ]; do
  sleep 300
done
log "Phase 4 marker found; proceeding with Phases 5-9."

# ---------------------------------------------------------------------------
# Phase 5: W11 within-aggregator diagnostic.
# ---------------------------------------------------------------------------
log "--- Phase 5: W11 within-aggregator (alpha=0, sigma=0) ---"
OUT="runs/cifar10_wag"
CFG="configs/cifar10_wag.yaml"
python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['aggregator']['alpha'] = 0.0
cfg['aggregator']['server_sigma'] = 0.0
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_wag'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
mkdir -p "$OUT"
if [ ! -e "$OUT/retrain_excl_c3" ]; then
  ln -s "$(cd "$(dirname "$MAIN_RETRAIN")" && pwd)/$(basename "$MAIN_RETRAIN")" "$OUT/retrain_excl_c3"
  log "[W11] symlinked retrain_excl_c3 from main run"
fi
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    --methods ours retrain no_unlearn 2>&1 | tee "$LOGDIR/phase5_W11_wag.log"
log "--- Phase 5 done ---"

# ---------------------------------------------------------------------------
# Phase 6: W12 near-IID CIFAR-10 (replay proxy).
# ---------------------------------------------------------------------------
log "--- Phase 6: W12 near-IID CIFAR-10 (dirichlet_alpha=10) ---"
OUT="runs/cifar10_iid"
CFG="configs/cifar10_iid.yaml"
python3 -c "
import yaml
with open('$BASE_CFG') as f: cfg = yaml.safe_load(f)
cfg['dataset']['dirichlet_alpha'] = 10.0
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['experiment']['name'] = 'cifar10_iid'
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase6_W12_iid.log"
log "--- Phase 6 done ---"

# ---------------------------------------------------------------------------
# Phase 7: W6 CIFAR-100 full benchmark.
# ---------------------------------------------------------------------------
log "--- Phase 7: W6 CIFAR-100 full benchmark ---"
CFG="configs/cifar100_w6.yaml"
OUT="runs/cifar100_w6"
python3 -c "
import yaml
with open('configs/cifar100_combined.yaml') as f: cfg = yaml.safe_load(f)
cfg['experiment']['device'] = 'auto'
cfg['experiment']['name'] = 'cifar100_w6'
cfg['federated']['clip_norm'] = 50.0
cfg['aggregator']['server_sigma'] = 0.001
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['baselines'] = {
    'halimi_ascent_steps': 50,
    'halimi_ascent_lr': 0.005,
    'halimi_radius': 3.0,
    'fedmemprune_quantile': 0.95,
    'fedmemprune_reset_scale': 0.0,
    'fedmemprune_include_head': True,
    'fedrecovery_weighting': 'linear',
    'fedrecovery_eta': 1.0,
    'fedrecovery_eps': 1.0,
    'fedrecovery_delta': 1.0e-5,
    'federaser_lr_calib': 0.5,
    'sifu_target_eps': 1.0,
    'participation_rho': 0.10,
    'history_stride': 5,
}
cfg['evaluation']['checkpoint_every'] = 1
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase7_W6_cifar100.log"
log "--- Phase 7 done ---"

# ---------------------------------------------------------------------------
# Phase 8: W6 Tiny-ImageNet full benchmark (longest; ~30-40h expected).
# ---------------------------------------------------------------------------
log "--- Phase 8: W6 Tiny-ImageNet full benchmark ---"
CFG="configs/tiny_imagenet_w6.yaml"
OUT="runs/tiny_imagenet_w6"
python3 -c "
import yaml
with open('configs/tiny_imagenet_combined.yaml') as f: cfg = yaml.safe_load(f)
cfg['experiment']['device'] = 'auto'
cfg['experiment']['name'] = 'tiny_imagenet_w6'
cfg['federated']['clip_norm'] = 50.0
cfg['aggregator']['server_sigma'] = 0.001
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['baselines'] = {
    'halimi_ascent_steps': 50,
    'halimi_ascent_lr': 0.005,
    'halimi_radius': 3.0,
    'fedmemprune_quantile': 0.95,
    'fedmemprune_reset_scale': 0.0,
    'fedmemprune_include_head': True,
    'fedrecovery_weighting': 'linear',
    'fedrecovery_eta': 1.0,
    'fedrecovery_eps': 1.0,
    'fedrecovery_delta': 1.0e-5,
    'federaser_lr_calib': 0.5,
    'sifu_target_eps': 1.0,
    'participation_rho': 0.10,
    'history_stride': 5,
}
cfg['evaluation']['checkpoint_every'] = 1
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase8_W6_tiny_imagenet.log"
log "--- Phase 8 done ---"

# ---------------------------------------------------------------------------
# Phase 9: W6 20 Newsgroups full benchmark.
# ---------------------------------------------------------------------------
log "--- Phase 9: W6 20 Newsgroups full benchmark ---"
CFG="configs/newsgroups20_w6.yaml"
OUT="runs/newsgroups20_w6"
python3 -c "
import yaml
with open('configs/newsgroups20_combined.yaml') as f: cfg = yaml.safe_load(f)
cfg['experiment']['device'] = 'auto'
cfg['experiment']['name'] = 'newsgroups20_w6'
cfg['output']['checkpoint_dir'] = '$OUT'
cfg['baselines'] = {
    'halimi_ascent_steps': 50,
    'halimi_ascent_lr': 0.005,
    'halimi_radius': 3.0,
    'fedmemprune_quantile': 0.95,
    'fedmemprune_reset_scale': 0.0,
    'fedmemprune_include_head': True,
    'fedrecovery_weighting': 'linear',
    'fedrecovery_eta': 1.0,
    'fedrecovery_eps': 1.0,
    'fedrecovery_delta': 1.0e-5,
    'federaser_lr_calib': 0.5,
    'sifu_target_eps': 1.0,
    'participation_rho': 0.10,
    'history_stride': 5,
}
cfg['evaluation']['checkpoint_every'] = 1
with open('$CFG', 'w') as f: yaml.safe_dump(cfg, f)
"
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase9_W6_newsgroups20.log"
log "--- Phase 9 done ---"

log "=== Queued Phases 5-9 complete ==="
