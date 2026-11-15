"""Shared helpers for baseline implementations.

- `BaselineResult`: uniform return type from every baseline's `unlearn` fn.
- `train_fedavg_and_maybe_store_history`: base training that (a) matches our
  method's training-time substrate for non-history baselines (Halimi,
  FedMemPrune) and (b) stores per-round update deltas for history-based
  baselines (FedEraser, SIFU, FedRecovery).
"""
from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..federated.aggregator import DecayedAggregator, DecayState
from ..federated.client import Client, ClientUpdate
from ..federated.checkpointing import ResumableSnapshot, RngState, SignalHandler, resume_state, save_snapshot
from ..federated.loop import _current_task_index, _sample_clients, cpu_state_dict
from ..federated.params import ParamDict, add_state, scale_state, zero_state_like
from ..federated.server import Server


logger = logging.getLogger("fu")


@dataclass
class BaselineResult:
    """Uniform return type from a baseline's unlearn function."""
    method: str
    unlearned_state: ParamDict
    wall_time_s: float
    guarantee: dict = field(default_factory=dict)     # {'eps': ..., 'delta': ...} or {}
    extras: dict = field(default_factory=dict)         # per-baseline metadata


def train_fedavg_and_maybe_store_history(
    *,
    model_factory: Callable[[], nn.Module],
    fcl_split,
    total_rounds: int,
    rounds_per_task: int,
    client_sample_frac: float,
    local_epochs: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    device: str,
    checkpoint_dir: Path,
    seed: int = 0,
    store_history: bool = False,
    history_stride: int = 1,
    evaluate_every: int = 25,
    evaluate_fn: Callable | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
) -> dict:
    """Vanilla FedAvg training with optional per-round update-history storage.

    Returns a dict with keys:
        global_state, round_index, history (list of {'round','deltas': {cid: delta}}),
        client_participation (list of client-round tuples).
    History is stored every `history_stride` rounds. Set `store_history=False`
    for Halimi/FedMemPrune to save memory; set True for the three history-based
    baselines. All state is periodically snapshotted for resume.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    initial_model = model_factory().to(device)
    global_state = cpu_state_dict(initial_model)
    # Reuse the DecayedAggregator with alpha very close to 0 so it acts as
    # FedAvg (weighted mean, no decay carry). We do not add server noise here.
    agg = DecayedAggregator(alpha=1e-6, server_lr=1.0, server_sigma=0.0, seed=seed)
    server = Server(
        global_state=global_state,
        decay_state=DecayState.zeros_from(global_state, alpha=1e-6),
        aggregator=agg,
    )

    participation: list[tuple[int, int]] = []  # (round, client_id)
    start_round = 0

    # History is stored append-only under `history_dir/r{N}.pt`. resume.pt
    # only holds the training state (small) — this keeps snapshot cost O(|θ|),
    # not O(N × R × |θ|). To reconstruct the full history list, call
    # `_load_history(history_dir)` below.
    history_dir = checkpoint_dir / "history"
    if store_history:
        history_dir.mkdir(parents=True, exist_ok=True)

    def _load_history_list() -> list[dict]:
        out = []
        if not history_dir.exists():
            return out
        for p in sorted(history_dir.glob("r*.pt"), key=lambda q: int(q.stem[1:])):
            out.append(torch.load(p, map_location="cpu", weights_only=False))
        return out

    # ---- Resume ------------------------------------------------------
    if resume:
        snap = resume_state(checkpoint_dir)
        if snap is not None:
            logger.info(f"[baseline] resuming from round={snap.round_index}")
            server.global_state = snap.global_state
            server.round_index = snap.round_index
            snap.rng_state.restore()
            rng = np.random.default_rng(seed)
            for _ in range(snap.round_index):
                _sample_clients(0, fcl_split.num_clients, client_sample_frac, rng)
            participation = snap.extras.get("participation", [])
            start_round = snap.round_index

    def _flush():
        snap = ResumableSnapshot(
            round_index=server.round_index,
            global_state=server.global_state,
            decay_tilde_delta=server.decay_state.tilde_delta,
            decay_alpha=server.decay_state.alpha,
            aggregator_gen_state=server.aggregator.gen.get_state(),
            summaries={},
            rng_state=RngState.capture(),
            extras={"participation": participation},
        )
        save_snapshot(snap, checkpoint_dir / "resume.pt")

    with SignalHandler(save_callback=_flush):
        for r in range(start_round, total_rounds):
            task_id = _current_task_index(r, rounds_per_task, fcl_split.num_tasks)
            sampled = _sample_clients(r, fcl_split.num_clients, client_sample_frac, rng)
            logger.info(f"[baseline] round={r} task={task_id} sampled={sampled}")
            round_updates: dict[int, ParamDict] = {}
            for cid in sampled:
                ds = fcl_split.client_dataset(cid, task_id=task_id)
                if len(ds) == 0:
                    continue
                client = Client(
                    client_id=cid, dataset=ds, local_epochs=local_epochs,
                    batch_size=batch_size, lr=lr, clip_norm=clip_norm, device=device,
                    text_mode=(fcl_split.modality == "text"),
                )
                model = model_factory().to(device)
                update = client.local_update(server.global_state, model, round_index=r)
                round_updates[cid] = update.delta
                participation.append((r, cid))
            # Sample-weighted mean, then apply.
            if round_updates:
                total_n = sum(1 for _ in round_updates)  # weight = 1/n (simple)
                mean_delta = zero_state_like(server.global_state)
                for cid, d in round_updates.items():
                    mean_delta = add_state(mean_delta, d, scale=1.0 / total_n)
                server.global_state = add_state(server.global_state, mean_delta, scale=1.0)
                if store_history and r % history_stride == 0:
                    entry = {"round": r, "deltas": {cid: d for cid, d in round_updates.items()}}
                    torch.save(entry, history_dir / f"r{r}.pt")
            server.round_index += 1

            if evaluate_fn is not None and (r + 1) % evaluate_every == 0:
                metrics = evaluate_fn(server, fcl_split, task_id=task_id)
                logger.info(f"[baseline] round={r+1} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            # Numbered archival checkpoints so SIFU can pick a rollback state.
            if (r + 1) % evaluate_every == 0:
                torch.save(
                    {"round": r + 1, "global_state": server.global_state},
                    checkpoint_dir / f"ckpt_r{r+1}.pt",
                )
            if (r + 1) % checkpoint_every == 0:
                _flush()

        _flush()  # final

    # NB: we deliberately do NOT eagerly load the full history list here —
    # it can be O(N × R × |θ|) bytes (e.g. 27 GB for ResNet-18 × 10 clients
    # × 40 history snapshots) and would cause the final return to swap for
    # tens of minutes on M4 unified memory. Downstream readers (run_baseline.py)
    # load the per-round files lazily off disk via `history_dir`.
    return {
        "global_state": server.global_state,
        "round_index": server.round_index,
        "history_dir": str(history_dir) if store_history else None,
        "participation": participation,
    }
