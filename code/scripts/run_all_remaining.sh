#!/usr/bin/env bash
# Chained orchestration of the remaining FORGE experiments.
#
# Phase 1: Ablations (alpha, sigma, summary type) on CIFAR-10 seed 0.
# Phase 2: LiRA n_shadow=32 for CIFAR-10 seeds 1 and 2.
# Phase 3: Full-budget FUIA (500 samples x 1000 steps) for CIFAR-10 seed 0.
# Phase 4: Re-aggregate the three-seed CSV.
# Phase 5: W11 within-aggregator (FORGE trained with alpha=0, sigma=0).
# Phase 6: W12 near-IID CIFAR-10 (dirichlet_alpha=10) proxy for replay.
# Phase 7: W6 CIFAR-100 full benchmark.
# Phase 8: W6 Tiny-ImageNet full benchmark.
# Phase 9: W6 20 Newsgroups full benchmark.
#
# Each phase is checkpointed via --skip-existing so interruption is safe.
# Logs land in runs/_remaining/<phase>.log.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="runs/_remaining"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

log "=== Starting run_all_remaining.sh ==="
log "ROOT=$ROOT"
log "Phases: 1=ablations, 2=lira32 s1/s2, 3=fuia_full, 4=re-aggregate,"
log "        5=W11 within-aggregator, 6=W12 near-IID, 7=W6 CIFAR-100,"
log "        8=W6 Tiny-ImageNet, 9=W6 20Newsgroups"

MAIN_RETRAIN="runs/cifar10_real/retrain_excl_c3"
if [ ! -d "$MAIN_RETRAIN" ]; then
  log "FATAL: main retrain dir $MAIN_RETRAIN not found; aborting."
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: Ablations (alpha, sigma, summary type).
# ---------------------------------------------------------------------------
log "--- Phase 1: ablations ---"
BASE_CFG="configs/cifar10_real.yaml"

run_abl() {
  local tag="$1"; local cfg="$2"; local out="$3"
  mkdir -p "$out"
  if [ ! -e "$out/retrain_excl_c3" ]; then
    ln -s "$(cd "$(dirname "$MAIN_RETRAIN")" && pwd)/$(basename "$MAIN_RETRAIN")" "$out/retrain_excl_c3"
    log "[abl] $tag: symlinked retrain_excl_c3 from main run"
  fi
  log "[abl] $tag -> $out"
  python3 scripts/benchmark.py --config "$cfg" --forget-client 3 --skip-existing \
      --methods ours retrain no_unlearn 2>&1 | tee -a "$LOGDIR/phase1_${tag}.log"
}

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
  run_abl "alpha_${alpha}" "$CFG" "$OUT"
done

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
  run_abl "sigma_${sigma}" "$CFG" "$OUT"
done

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
  run_abl "summary_${summ}" "$CFG" "$OUT"
done

log "--- Phase 1 done ---"

# ---------------------------------------------------------------------------
# Phase 2: LiRA n_shadow=32 for CIFAR-10 seeds 1 and 2.
# ---------------------------------------------------------------------------
log "--- Phase 2: LiRA-32 for seeds 1 and 2 ---"

for seed in 1 2; do
  OUT="runs/cifar10_s${seed}"
  log "[lira32] seed=$seed"
  rm -f "$OUT/benchmark_c3.csv" "$OUT/benchmark_c3.json"
  python3 scripts/benchmark.py --config "configs/cifar10_s${seed}.yaml" \
      --forget-client 3 --skip-existing --run-lira --lira-n-shadow 32 \
      2>&1 | tee "$LOGDIR/phase2_s${seed}.log"
done

log "--- Phase 2 done ---"

# ---------------------------------------------------------------------------
# Phase 3: Full-budget FUIA (500 x 1000) for CIFAR-10 seed 0.
# ---------------------------------------------------------------------------
log "--- Phase 3: full-budget FUIA (500 samples x 1000 steps) ---"
rm -f runs/cifar10_real/fuia_c3.json
python3 scripts/benchmark.py --config configs/cifar10_real.yaml --forget-client 3 \
    --skip-existing --run-fuia --fuia-n-samples 500 --fuia-steps 1000 \
    --methods ours retrain no_unlearn fedrecovery fedrecovery_eps_inf \
    2>&1 | tee "$LOGDIR/phase3_fuia_full.log"
log "--- Phase 3 done ---"

# ---------------------------------------------------------------------------
# Phase 4: Re-aggregate three-seed CSV.
# ---------------------------------------------------------------------------
log "--- Phase 4: re-aggregate three-seed CSV ---"
python3 scripts/aggregate_results.py \
    runs/cifar10_real/benchmark_c3.json \
    runs/cifar10_s1/benchmark_c3.json \
    runs/cifar10_s2/benchmark_c3.json \
    --out runs/aggregated_3seed_v4.csv 2>&1 | tee "$LOGDIR/phase4_aggregate.log"
log "--- Phase 4 done ---"

# ---------------------------------------------------------------------------
# Phase 5: W11 within-aggregator diagnostic.
# FORGE trained with alpha=0, server_sigma=0 isolates the aggregator's
# utility cost from the erasure cost. Retrain is symlinked from main run.
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
# Phase 6: W12 near-IID proxy for replay.
# Setting dirichlet_alpha=10 approaches an IID partition, which lifts
# absolute utility and tests whether the FORGE/baseline ordering holds at
# a realistic FL operating point. Retrain and shared base must be
# re-trained since the partition changes.
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
# Patch the *_combined.yaml: device cuda -> auto, add baselines block,
# and relax clip_norm so ResNet-18 is trainable.
# ---------------------------------------------------------------------------
log "--- Phase 7: W6 CIFAR-100 full benchmark ---"
CFG="configs/cifar100_w6.yaml"
OUT="runs/cifar100_w6"
python3 -c "
import yaml
with open('configs/cifar100_combined.yaml') as f: cfg = yaml.safe_load(f)
cfg['experiment']['device'] = 'auto'
cfg['experiment']['name'] = 'cifar100_w6'
cfg['federated']['clip_norm'] = 50.0  # match CIFAR-10 trainable regime
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
# Phase 8: W6 Tiny-ImageNet full benchmark.
# Longest of the three (~30-40h expected on M4).
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
# Text benchmark on distilbert; expect ~10-15h.
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

log "=== All remaining experiments complete ==="
log "Outputs:"
log "  runs/cifar10_abl_*         : ablation benchmarks (Phase 1)"
log "  runs/cifar10_s{1,2}/       : LiRA-32 upgraded seeds (Phase 2)"
log "  runs/cifar10_real/fuia_c3.json : full-budget FUIA (Phase 3)"
log "  runs/aggregated_3seed_v4.csv   : refreshed three-seed table (Phase 4)"
log "  runs/cifar10_wag/          : W11 within-aggregator (Phase 5)"
log "  runs/cifar10_iid/          : W12 near-IID proxy (Phase 6)"
log "  runs/cifar100_w6/          : W6 CIFAR-100 (Phase 7)"
log "  runs/tiny_imagenet_w6/     : W6 Tiny-ImageNet (Phase 8)"
log "  runs/newsgroups20_w6/      : W6 20 Newsgroups (Phase 9)"
