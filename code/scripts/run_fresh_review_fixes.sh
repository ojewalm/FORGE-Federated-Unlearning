#!/usr/bin/env bash
# Chain for the 3 compute items from the fresh-review:
#   (1) LiRA n_shadow=128 on primary non-IID stationary seed 0
#       (back up n=32 output first; this overwrites with n=128)
#   (3) SIFU + FedRec_eps_inf on primary non-IID stationary, 3 seeds
#   (4) CIFAR-100 non-IID stationary seed 1 (retrain + no_unlearn + ours + LiRA n=32)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=runs/_remaining
mkdir -p "$LOG"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orchestrator.log"; }

# -------- (1) LiRA n=128 on primary non-IID stationary seed 0 --------
log "=== (1) LiRA n=128 on primary non-IID seed 0 starting ==="
# Preserve the existing n=32 output.
if [ -f runs/cifar10_noniid/benchmark_c3.json ]; then
    cp -n runs/cifar10_noniid/benchmark_c3.json runs/cifar10_noniid/benchmark_c3_lira32.json || true
    rm -f runs/cifar10_noniid/benchmark_c3.json runs/cifar10_noniid/benchmark_c3.csv
fi
python3 scripts/benchmark.py \
    --config configs/cifar10_noniid.yaml --forget-client 3 \
    --methods retrain no_unlearn ours --skip-existing \
    --run-lira --lira-n-shadow 128 \
    2>&1 | tee "$LOG/noniid_lira128_s0.log"
log "=== (1) LiRA n=128 primary seed 0 done (exit=${PIPESTATUS[0]}) ==="

# -------- (4) CIFAR-100 non-IID stationary seed 1 --------
log "=== (4) CIFAR-100 non-IID seed 1 starting ==="
python3 scripts/benchmark.py \
    --config configs/cifar100_noniid_s1.yaml --forget-client 3 \
    --methods retrain no_unlearn ours \
    --run-lira --lira-n-shadow 32 \
    2>&1 | tee "$LOG/cifar100_noniid_s1.log"
log "=== (4) CIFAR-100 non-IID seed 1 done (exit=${PIPESTATUS[0]}) ==="

# -------- (3) SIFU + FedRec_eps_inf on primary non-IID, 3 seeds --------
# Primary non-IID training checkpoints already exist; baselines train the shared
# FedAvg base (common across SIFU and FedRec) then run their erasure.
for S in "" "_s1" "_s2"; do
    CFG="configs/cifar10_noniid${S}.yaml"
    log "=== (3) SIFU + FedRec_eps_inf on cifar10_noniid${S} starting ==="
    python3 scripts/benchmark.py \
        --config "$CFG" --forget-client 3 \
        --methods sifu fedrecovery_eps_inf retrain no_unlearn --skip-existing \
        2>&1 | tee "$LOG/noniid_baselines${S}.log"
    log "=== (3) baselines cifar10_noniid${S} done (exit=${PIPESTATUS[0]}) ==="
done

log "=== all fresh-review fixes done ==="
