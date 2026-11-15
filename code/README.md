# FORGE, reference implementation

Reference implementation of the FORGE federated continual unlearning protocol
used to produce every table, figure, and appendix result in the companion paper.

## Layout

```
code/
├── configs/           YAML experiment configs (one per benchmark x seed).
├── src/
│   ├── datasets/      CIFAR-10, CIFAR-100, FEMNIST, Tiny-ImageNet, 20 Newsgroups
│   │                  with task-incremental, non-IID stationary, and replay splits.
│   ├── models/        ResNet-18 (small-input stem), small CNN, DistilBERT.
│   ├── federated/     Server, decayed aggregator, EMA state, clipping,
│   │                  per-round Gaussian noise, resumable checkpointing.
│   ├── unlearning/    Pre-committed summary channel, non-interactive head
│   │                  subtraction, forward-forgetting rate, adaptive GDP
│   │                  composition.
│   ├── attacks/       Yeom threshold MIA, shared-shadow LiRA, FUIA gradient
│   │                  inversion.
│   ├── baselines/     FedEraser, SIFU (with proper fine-tuning replay),
│   │                  FedRecovery (epsilon-sweep), Halimi, FedMemPrune.
│   ├── metrics/       Accuracy, Yeom/LiRA advantage, retrain-KL, FUIA PSNR.
│   └── utils/         Config, logging, deterministic seeding, Pedersen
│                      commitment helpers, hash transcript.
├── scripts/
│   ├── train.py              Federated training with FORGE.
│   ├── retrain.py            Vanilla-FedAvg retrain from scratch (gold).
│   ├── unlearn.py            Non-interactive erasure on a trained checkpoint.
│   ├── eval.py               Metric evaluation for a (pre, post) pair.
│   ├── benchmark.py          End-to-end: trains, unlearns, evaluates, and
│   │                         scores LiRA / FUIA for a method list.
│   ├── run_noniid_seeds_12.sh   Primary 3-seed CIFAR-10 non-IID driver.
│   ├── run_ablations.sh         Alpha / sigma / summary-type sweep.
│   └── measure_lipschitz.py  Empirical per-round Lipschitz probe.
└── tests/             Sanity tests (deterministic resume, GDP composition).
```

## Quick start

```bash
cd code
pip install -r requirements.txt
```

Datasets download on first use. All runs are deterministic given a fixed seed
and support interruption / resume via `resume.pt` written every round.

## Reproducing the primary benchmark (CIFAR-10 non-IID stationary, 3 seeds)

```bash
bash scripts/run_noniid_seeds_12.sh
```

Approximately 60 hours on an Apple M4 including LiRA at `n_shadow=32`. Writes
`runs/cifar10_noniid{,_s1,_s2}/benchmark_c3.json`, which populates Table 5 of
the paper.

## Reproducing the class-incremental stress test (12 methods, 3 seeds)

```bash
python3 scripts/benchmark.py --config configs/cifar10_real.yaml \
    --forget-client 3 --run-lira --lira-n-shadow 128
# repeat for configs/cifar10_s1.yaml and cifar10_s2.yaml
```

Approximately 5 hours per seed. Writes `runs/cifar10_real/benchmark_c3.json`,
which populates Table 6 and Appendix E.

## Reproducing ablation, cross-dataset, and memorization-regime appendices

```bash
# Hyperparameter ablation on the primary protocol (Appendix F).
bash scripts/run_noniid_reframe.sh

# CIFAR-100 non-IID stationary cross-dataset (Appendix G).
python3 scripts/benchmark.py --config configs/cifar100_noniid.yaml \
    --forget-client 3 --run-lira --lira-n-shadow 32

# FEMNIST cross-dataset (Section 7.8).
python3 scripts/benchmark.py --config configs/femnist.yaml --forget-client 3

# Memorization-regime sensitivity (Appendix J).
bash scripts/run_fewclients.sh
```

## Reproducing the non-convex Lipschitz probe

```bash
python3 scripts/measure_lipschitz.py \
    --config configs/cifar10_noniid.yaml \
    --checkpoint runs/cifar10_noniid/final.pt \
    --n-clients 8 --n-directions 4 --epsilon 1e-3 --seed 0
```

Reproduces the `L' <= 88` finding of Appendix A.3 at ResNet-18 scale.

## Hardware and reproducibility

All runs reported in the paper used an Apple M4 (MPS backend) with PyTorch 2.8.
Seeds are fixed, dataset partitioning is deterministic, and the checkpointing
mechanism supports interruption and resume so the full benchmark completes on
consumer hardware without supervision.
