#!/usr/bin/env bash
# Sequentially run non-IID stationary CIFAR-10 benchmark for seeds 1 and 2.
# Mirrors the seed-0 invocation: retrain + no_unlearn + ours + LiRA n=32.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="runs/_remaining"
mkdir -p "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

for S in 1 2; do
  log "=== non-IID stationary CIFAR-10 seed $S starting ==="
  python3 scripts/benchmark.py \
      --config "configs/cifar10_noniid_s${S}.yaml" \
      --forget-client 3 \
      --methods retrain no_unlearn ours \
      --run-lira --lira-n-shadow 32 \
      2>&1 | tee "$LOGDIR/noniid_s${S}.log"
  rc=${PIPESTATUS[0]}
  log "=== non-IID stationary CIFAR-10 seed $S exit=$rc ==="
done

log "=== all seeds done ==="
