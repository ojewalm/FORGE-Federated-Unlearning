#!/usr/bin/env bash
# Chained runs for the Option-2 reframe:
#   (A) CIFAR-100 non-IID stationary cross-dataset (seed 0, retrain+no_unlearn+ours+LiRA n=32)
#   (B) Ablation sweep (6 cells) on non-IID stationary CIFAR-10
#       alpha ∈ {0.1, 0.9}, sigma ∈ {1e-4, 1e-2}, summary ∈ {logits},
#       within-aggregator (alpha=0, sigma=0)
#   Shared seed-0 non-IID retrain is pre-copied into each ablation cell
#   so --skip-existing avoids re-running retrain 6×.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=runs/_remaining
mkdir -p "$LOG"
BASE_RETRAIN=runs/cifar10_noniid/retrain_excl_c3/final.pt
BASE_CFG=configs/cifar10_noniid.yaml

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orchestrator.log"; }

# ----- (A) CIFAR-100 non-IID stationary -----
log "=== (A) CIFAR-100 non-IID stationary starting ==="
python3 scripts/benchmark.py \
    --config configs/cifar100_noniid.yaml \
    --forget-client 3 \
    --methods retrain no_unlearn ours \
    --run-lira --lira-n-shadow 32 \
    2>&1 | tee "$LOG/cifar100_noniid.log"
log "=== (A) CIFAR-100 non-IID stationary done (exit=${PIPESTATUS[0]}) ==="

# ----- (B) Ablation sweep -----
# Each cell inherits from cifar10_noniid.yaml and overrides one hyperparameter.
run_cell() {
    local name="$1"; shift
    local overrides="$1"; shift
    local out="runs/cifar10_abl_${name}"
    local cfg="configs/cifar10_abl_${name}.yaml"
    log "--- ablation cell ${name} starting ---"
    python3 - <<PY
import yaml
with open("$BASE_CFG") as f: c = yaml.safe_load(f)
${overrides}
c['experiment']['name'] = "cifar10_abl_${name}"
c['output']['checkpoint_dir'] = "$out"
with open("$cfg", "w") as f: yaml.safe_dump(c, f)
PY
    mkdir -p "$out/retrain_excl_c3"
    [ -f "$BASE_RETRAIN" ] && cp -n "$BASE_RETRAIN" "$out/retrain_excl_c3/final.pt" || true
    python3 scripts/benchmark.py --config "$cfg" --forget-client 3 \
        --methods ours retrain no_unlearn --skip-existing \
        2>&1 | tee "$LOG/abl_${name}.log"
    log "--- ablation cell ${name} done (exit=${PIPESTATUS[0]}) ---"
}

run_cell "alpha01"       "c['aggregator']['alpha']=0.1"
run_cell "alpha09"       "c['aggregator']['alpha']=0.9"
run_cell "sigma1e4"      "c['aggregator']['server_sigma']=0.0001"
run_cell "sigma1e2"      "c['aggregator']['server_sigma']=0.01"
run_cell "summary_logits" "c['summary']['type']='logits'"
run_cell "substrate"     "c['aggregator']['alpha']=0; c['aggregator']['server_sigma']=0"

log "=== all reframe runs done ==="
