#!/usr/bin/env bash
# Re-queue of the four phases that failed in run_queued_W6_W11_W12.sh:
#   Phase 5 (W11 within-aggregator): aggregator rejected alpha=0; patched.
#   Phase 7 (W6 CIFAR-100): logits-probe shape mismatch; patched.
#   Phase 8 (W6 Tiny-ImageNet): dataset missing; downloaded.
#   Phase 9 (W6 20 Newsgroups): logits-probe shape mismatch; patched.
#
# For 7 and 9, --skip-existing reuses the already-complete training; only
# the unlearn + evaluation re-runs (minutes rather than hours).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="runs/_remaining"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

MAIN_RETRAIN="runs/cifar10_real/retrain_excl_c3"
BASE_CFG="configs/cifar10_real.yaml"

log "=== run_requeue.sh starting ==="

# ---------------------------------------------------------------------------
# Re-run Phase 5: W11 within-aggregator.
# Config has alpha=0.0; aggregator patch now allows it.
# Clear old partial state so training restarts cleanly.
# ---------------------------------------------------------------------------
log "--- Re-run Phase 5: W11 within-aggregator ---"
OUT="runs/cifar10_wag"
CFG="configs/cifar10_wag.yaml"
rm -rf "$OUT"
mkdir -p "$OUT"
ln -sf "$(cd "$(dirname "$MAIN_RETRAIN")" && pwd)/$(basename "$MAIN_RETRAIN")" "$OUT/retrain_excl_c3"
python3 scripts/benchmark.py --config "$CFG" --forget-client 3 --skip-existing \
    --methods ours retrain no_unlearn 2>&1 | tee "$LOGDIR/phase5_W11_wag_requeue.log"
log "--- Phase 5 re-run done ---"

# ---------------------------------------------------------------------------
# Re-run Phase 7: W6 CIFAR-100.
# Training is already done (runs/cifar100_w6/final.pt exists); only the
# unlearn step and the baselines pass need to run. Clear the stale
# benchmark_c3.* so the final table rebuilds.
# ---------------------------------------------------------------------------
log "--- Re-run Phase 7: W6 CIFAR-100 (unlearn + baselines only) ---"
rm -f runs/cifar100_w6/benchmark_c3.csv runs/cifar100_w6/benchmark_c3.json \
      runs/cifar100_w6/final_unlearned_c3.pt
python3 scripts/benchmark.py --config configs/cifar100_w6.yaml --forget-client 3 \
    --skip-existing 2>&1 | tee "$LOGDIR/phase7_W6_cifar100_requeue.log"
log "--- Phase 7 re-run done ---"

# ---------------------------------------------------------------------------
# Re-run Phase 9: W6 20 Newsgroups.
# Same situation: training already done; only unlearn + baselines rerun.
# ---------------------------------------------------------------------------
log "--- Re-run Phase 9: W6 20 Newsgroups (unlearn + baselines only) ---"
rm -f runs/newsgroups20_w6/benchmark_c3.csv runs/newsgroups20_w6/benchmark_c3.json \
      runs/newsgroups20_w6/final_unlearned_c3.pt
python3 scripts/benchmark.py --config configs/newsgroups20_w6.yaml --forget-client 3 \
    --skip-existing 2>&1 | tee "$LOGDIR/phase9_W6_newsgroups20_requeue.log"
log "--- Phase 9 re-run done ---"

# ---------------------------------------------------------------------------
# Re-run Phase 8: W6 Tiny-ImageNet (longest; dataset now downloaded).
# Scheduled last because it is the longest run (~30-40h).
# ---------------------------------------------------------------------------
log "--- Re-run Phase 8: W6 Tiny-ImageNet (full training + unlearn) ---"
python3 scripts/benchmark.py --config configs/tiny_imagenet_w6.yaml --forget-client 3 \
    --skip-existing 2>&1 | tee "$LOGDIR/phase8_W6_tiny_imagenet_requeue.log"
log "--- Phase 8 re-run done ---"

log "=== run_requeue.sh complete ==="
