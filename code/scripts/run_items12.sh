#!/usr/bin/env bash
# Items 1 and 2 from the latest reviewer-recommendation pass:
#   Item 1: cross-device scale (1000 clients, 2% sample fraction) on CIFAR-10.
#   Item 2: replay-enabled CIFAR-10 (per-client FIFO buffer, capacity 40).
# Both use the main-text FORGE training protocol (ResNet-18, 200 rounds).
# Runs the three preserved-utility methods only (ours, retrain, no_unlearn)
# plus SIFU, to keep wall-clock down.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="runs/_remaining"
mkdir -p "$LOGDIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

log "=== run_items12.sh starting ==="

# ---- Item 2: Replay-enabled CIFAR-10 ----
log "--- Item 2: replay-enabled CIFAR-10 (capacity=40, admit=0.5) ---"
python3 scripts/benchmark.py --config configs/cifar10_replay.yaml --forget-client 3 --skip-existing \
    --methods ours retrain no_unlearn sifu 2>&1 | tee "$LOGDIR/item2_replay.log"
log "--- Item 2 done ---"

# ---- Item 1: Cross-device CIFAR-10 (1000 clients) ----
log "--- Item 1: cross-device CIFAR-10 (1000 clients, 2% sample) ---"
python3 scripts/benchmark.py --config configs/cifar10_crossdev.yaml --forget-client 3 --skip-existing \
    --methods ours retrain no_unlearn sifu 2>&1 | tee "$LOGDIR/item1_crossdev.log"
log "--- Item 1 done ---"

log "=== run_items12.sh complete ==="
