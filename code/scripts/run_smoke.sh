#!/usr/bin/env bash
# End-to-end smoke sequence: train → retrain (excluding one client) → unlearn → summarise.
# Assumes CIFAR-100 is already prepared under ./data.
set -euo pipefail

CFG="${CFG:-configs/cifar100_smoke.yaml}"
CLIENT="${CLIENT:-3}"
R_FORGET="${R_FORGET:-5}"
R_REQUEST="${R_REQUEST:-15}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/train.py    --config "$CFG"
python3 scripts/retrain.py  --config "$CFG" --exclude-client "$CLIENT"
python3 scripts/unlearn.py  --config "$CFG" \
        --checkpoint runs/cifar100_smoke/final.pt \
        --client "$CLIENT" \
        --r-forget "$R_FORGET" \
        --r-request "$R_REQUEST" \
        --output runs/cifar100_smoke/final_unlearned_c${CLIENT}.pt
python3 scripts/summarise_run.py \
        --config "$CFG" \
        --pre runs/cifar100_smoke/final.pt \
        --post runs/cifar100_smoke/final_unlearned_c${CLIENT}.pt \
        --retrain runs/cifar100_smoke/retrain_excl_c${CLIENT}/final.pt \
        --client "$CLIENT" \
        --output-json runs/cifar100_smoke/summary_c${CLIENT}.json

echo "Smoke sequence complete. Summary: runs/cifar100_smoke/summary_c${CLIENT}.json"
