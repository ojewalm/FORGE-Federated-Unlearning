# Forgetting on a Schedule — reference implementation

Reference implementation for the certified federated continual unlearning
protocol described in the companion paper (`../paper/`).

## Layout

```
code/
├── configs/           YAML experiment configs (one per dataset x method)
├── src/
│   ├── datasets/      CIFAR-100, FEMNIST, Tiny-ImageNet, 20 Newsgroups loaders
│   ├── models/        ResNet-18, small CNN, distilled-BERT
│   ├── federated/     Client + server, decayed aggregation, DP mechanisms
│   ├── unlearning/    Summary channel, non-interactive erasure
│   ├── attacks/       MIA (threshold + LiRA-lite), FUIA stub
│   ├── metrics/       Forgetting, utility, cost
│   └── utils/         Config, logging, seeding, checkpointing
├── scripts/
│   ├── train.py       Run federated continual training
│   ├── unlearn.py     Apply erasure procedure to a checkpoint
│   └── eval.py        Compute all metrics for a checkpoint pair
└── tests/             Sanity tests
```

## Quick start

```bash
cd code
pip install -r requirements.txt

# Download and cache datasets (idempotent)
python -m src.datasets.prepare --dataset cifar100
python -m src.datasets.prepare --dataset femnist
python -m src.datasets.prepare --dataset tiny_imagenet
python -m src.datasets.prepare --dataset newsgroups20

# Train
python scripts/train.py --config configs/cifar100_combined.yaml

# Unlearn a departed client
python scripts/unlearn.py \
    --checkpoint runs/cifar100_combined/ckpt_r200.pt \
    --client 42 \
    --config configs/cifar100_combined.yaml

# Evaluate
python scripts/eval.py \
    --pre runs/cifar100_combined/ckpt_r200.pt \
    --post runs/cifar100_combined/ckpt_r200_unlearned_c42.pt \
    --config configs/cifar100_combined.yaml
```

## Status

This is a **scaffold**. The federated training loop, dataset loaders, model
definitions, and metric implementations are functional. The theoretical
components (forward-forgetting rate computation, summary-subtraction step,
composition accounting) are implemented as first-pass versions and marked with
`# TODO(theory)` for tightening against the paper's proofs. Attacks (MIA, FUIA)
are implemented at the level needed for the reported metrics; production-grade
attack tuning is future work.
