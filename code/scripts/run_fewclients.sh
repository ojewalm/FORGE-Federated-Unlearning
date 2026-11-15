#!/usr/bin/env bash
# Stronger-memorization benchmark: 10 clients, full participation, same model/LR as
# the primary non-IID stationary config. Each client's per-round influence is 10x
# the primary (1/10 vs 1/100 of aggregation), expected to produce stronger
# membership-inference signal.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=runs/_remaining
mkdir -p "$LOG"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orchestrator.log"; }

log "=== fewclients benchmark starting ==="
# With only 10 clients, departing client 3 is 10% of the data pool.
python3 scripts/benchmark.py \
    --config configs/cifar10_fewclients.yaml --forget-client 3 \
    --methods retrain no_unlearn ours \
    --run-lira --lira-n-shadow 32 \
    2>&1 | tee "$LOG/fewclients.log"
log "=== fewclients benchmark done (exit=${PIPESTATUS[0]}) ==="
