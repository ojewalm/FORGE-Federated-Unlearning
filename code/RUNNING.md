# Running the real benchmark

Everything below assumes you're in `code/` and Python + deps are installed.

## Kick off

```bash
python3 scripts/benchmark.py --config configs/cifar100_real.yaml --forget-client 3 --skip-existing
```

The `--skip-existing` flag means any completed phase (retrain gold, ours train
+ unlearn, shared base, each baseline unlearn) is not re-run. Kill the process
at any time and re-run the same command — it picks up mid-phase.

## Phases (in this order)

| # | Phase | Script | Duration (est on M4) |
| --- | --- | --- | --- |
| 1 | Retrain-from-scratch excluding client i* | `retrain.py` | ~40 min |
| 2 | Our combined method (train) | `train.py` | ~40 min |
| 3 | Our combined method (unlearn) | `unlearn.py` | seconds |
| 4 | Shared FedAvg-with-history base | `train_shared_base.py` | ~40 min |
| 5 | Baseline unlearns (5×) | `run_baseline.py` | seconds each |
| 6 | Evaluate all methods | (inside `benchmark.py`) | ~2 min |

Total: ~2 hours end-to-end on M4 MPS (`configs/cifar100_real.yaml`, 200
rounds, ResNet-18, batch 32).

## Check status without disturbing the run

```bash
python3 scripts/status.py runs/cifar100_real
```

Prints one line per phase: `not started`, `in-progress: round=X saved=Xs ago`,
or `complete: <file>`.

## Resume after interrupt / sleep / crash

Every training round writes an atomic `resume.pt` snapshot into the phase's
checkpoint directory. On startup a script looks for `resume.pt` and picks up
from the recorded round with all RNG state restored. So:

```bash
# Was interrupted mid-training. Just re-run:
python3 scripts/benchmark.py --config configs/cifar100_real.yaml --forget-client 3 --skip-existing
```

`--skip-existing` skips completed phases (checked via their `final*.pt`), and
in-progress phases resume from `resume.pt`.

## Killing a run cleanly

```bash
# Ctrl-C sends SIGINT; the training loop flushes resume.pt before exiting.
kill -TERM <pid>    # also works; SIGTERM is caught the same way
```

Killing with SIGKILL (`kill -9`) skips the flush handler — you may lose the
last round of progress, but the most recent atomic `resume.pt` is still
readable.

## Reading results

When the benchmark completes:

- `runs/cifar100_real/benchmark_c3.csv` — one row per method with accuracy,
  MIA advantage, KL vs retrain, wall-clock, guarantee, storage.
- `runs/cifar100_real/benchmark_c3.json` — same data in JSON.

Individual method artefacts live under:

- `runs/cifar100_real/final.pt` — ours' pre-unlearn model
- `runs/cifar100_real/final_unlearned_c3.pt` — ours' post-unlearn model +
  certificate
- `runs/cifar100_real/retrain_excl_c3/final.pt` — retrain gold
- `runs/cifar100_real/shared_base/final_base.pt` — shared FedAvg base for
  baselines
- `runs/cifar100_real/shared_base/history.pt` — per-round per-client deltas
- `runs/cifar100_real/baseline_<method>_c3/final_unlearned_c3.pt` — each
  baseline's post-unlearn output

## Debugging a stuck run

```bash
# Are the training processes actually alive?
ps auxww | grep -E "scripts/(retrain|train|run_baseline|benchmark|train_shared_base)" | grep -v grep

# What round is each phase on?
python3 scripts/status.py runs/cifar100_real

# What's in the aggregate log?
tail -f runs/cifar100_real/fu*.log
tail -f runs/cifar100_real/retrain_excl_c3/fu-retrain.log
tail -f runs/cifar100_real/shared_base/fu-shared-base.log
```

## Scaling further

`configs/cifar100_combined.yaml` is the paper-scale config (500 rounds).
On M4 it will take ~5 hours per training, ~15 hours total. Use the same
command with `--config configs/cifar100_combined.yaml`.

For other datasets, use `configs/femnist_combined.yaml`,
`configs/tiny_imagenet_combined.yaml`, or `configs/newsgroups20_combined.yaml`
— note that Tiny-ImageNet and 20 Newsgroups may need `python -m
src.datasets.prepare --dataset X` first.
