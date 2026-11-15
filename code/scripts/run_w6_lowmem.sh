#!/usr/bin/env bash
# Low-memory re-run of W6 20 Newsgroups and Tiny-ImageNet.
# batch_size has been reduced to 8 in both configs.
# Runs sequentially (never both at once) to keep peak memory below ~5 GB.
# Clears partial 20News shared_base so it rebuilds from scratch.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOGDIR="runs/_remaining"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/orchestrator.log"; }

log "=== run_w6_lowmem.sh starting (batch_size=8) ==="

# Phase 9 retry: 20 Newsgroups.
log "--- 20 Newsgroups: clearing partial shared_base and re-running ---"
rm -rf runs/newsgroups20_w6/shared_base
rm -f runs/newsgroups20_w6/benchmark_c3.csv runs/newsgroups20_w6/benchmark_c3.json
python3 scripts/benchmark.py --config configs/newsgroups20_w6.yaml \
    --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase9_W6_newsgroups20_lowmem.log"
log "--- 20 Newsgroups done ---"

# Phase 8 retry: Tiny-ImageNet.
log "--- Tiny-ImageNet: full training + unlearn ---"
rm -rf runs/tiny_imagenet_w6
python3 scripts/benchmark.py --config configs/tiny_imagenet_w6.yaml \
    --forget-client 3 --skip-existing \
    2>&1 | tee "$LOGDIR/phase8_W6_tiny_imagenet_lowmem.log"
log "--- Tiny-ImageNet done ---"

log "=== run_w6_lowmem.sh complete ==="
