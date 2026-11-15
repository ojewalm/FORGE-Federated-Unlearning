# FORGE, artefact for anonymous review

Reference implementation for *FORGE: Federated Continual Unlearning with
Formal Guarantees*, submitted to PoPETs 2027.

## Reproducing the experiments

```bash
cd code
pip install -r requirements.txt
# Primary CIFAR-10 non-IID stationary benchmark (3 seeds, ~60 h on an Apple M4).
bash scripts/run_noniid_seeds_12.sh
# Class-incremental stress test with 11 baselines (~5 h per seed).
python3 scripts/benchmark.py --config configs/cifar10_real.yaml --forget-client 3 \
    --run-lira --lira-n-shadow 128
```

Datasets (CIFAR-10, CIFAR-100, FEMNIST) are downloaded on first use. All
benchmarks are deterministic given a fixed seed and support interruption and
resume via `resume.pt`.

## Anonymity note

This bundle contains no author identifiers, institutional affiliations, or
credentials. Previous run outputs (`runs/`), downloaded datasets (`data/`), and
version-control history have been stripped.
