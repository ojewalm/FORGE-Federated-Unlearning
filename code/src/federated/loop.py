"""Federated continual learning training loop (Algorithm 1 in the paper).

Interruptible + resumable: writes a `resume.pt` snapshot every
`checkpoint_every` rounds and on SIGTERM/SIGINT. On startup, if a `resume.pt`
exists, training picks up from the last completed round; otherwise it starts
fresh. Long M4 runs that span multiple sleep/wake cycles or user interrupts
can be resumed by re-running the same command.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .aggregator import DecayedAggregator
from .backdoor import BackdoorConfig, PoisonedDataset, should_inject
from .checkpointing import ResumableSnapshot, RngState, SignalHandler, resume_state, save_snapshot
from .client import Client
from .params import ParamDict
from .replay import ReplayBuffer
from .server import Server, commit_summary


logger = logging.getLogger("fu")


def cpu_state_dict(model: nn.Module) -> ParamDict:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def _sample_clients(round_index: int, num_clients: int, sample_frac: float, rng: np.random.Generator) -> list[int]:
    n = max(1, int(round(num_clients * sample_frac)))
    return rng.choice(num_clients, size=n, replace=False).tolist()


def _current_task_index(round_index: int, rounds_per_task: int, num_tasks: int) -> int:
    return min(round_index // rounds_per_task, num_tasks - 1)


def federated_train(
    *,
    server: Server,
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
    summarise_fn: Callable | None = None,
    summary_frequency: int = 20,
    summary_eps: float = 2.0,
    summary_delta: float = 1e-5,
    evaluate_every: int = 25,
    evaluate_fn: Callable | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 1,
    seed: int = 0,
    dataset_name: str = "",
    replay_buffer: ReplayBuffer | None = None,
    backdoor: BackdoorConfig | None = None,
) -> Server:
    """Run the FCL training loop with the decayed-aggregation protocol.

    Args:
        checkpoint_every: how often to write `resume.pt` (in rounds). Set to 1
            for maximum resilience; set higher to reduce I/O when I/O is slow.
    """
    rng = np.random.default_rng(seed)
    text_mode = fcl_split.modality == "text"

    # ---- Resume logic --------------------------------------------------
    start_round = 0
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        snap = resume_state(checkpoint_dir)
        if snap is not None:
            logger.info(f"Resuming from snapshot at round={snap.round_index} saved {snap.saved_at:.0f}s ago")
            server.global_state = snap.global_state
            server.decay_state.tilde_delta = snap.decay_tilde_delta
            server.decay_state.alpha = snap.decay_alpha
            server.aggregator.gen.set_state(snap.aggregator_gen_state)
            server.summaries = snap.summaries
            server.round_index = snap.round_index
            snap.rng_state.restore()
            # numpy rng comes back via rng_state but our client-sampler `rng`
            # was seeded independently — reconstruct it deterministically.
            rng = np.random.default_rng(seed)
            for _ in range(snap.round_index):
                _sample_clients(0, fcl_split.num_clients, client_sample_frac, rng)
            start_round = snap.round_index

    # ---- SIGTERM/SIGINT flush ------------------------------------------
    def _flush_snapshot():
        if checkpoint_dir is None:
            return
        snap = ResumableSnapshot(
            round_index=server.round_index,
            global_state=server.global_state,
            decay_tilde_delta=server.decay_state.tilde_delta,
            decay_alpha=server.decay_state.alpha,
            aggregator_gen_state=server.aggregator.gen.get_state(),
            summaries=server.summaries,
            rng_state=RngState.capture(),
            extras={"dataset": dataset_name, "total_rounds": total_rounds},
        )
        save_snapshot(snap, checkpoint_dir / "resume.pt")

    with SignalHandler(save_callback=_flush_snapshot):
        for r in range(start_round, total_rounds):
            task_id = _current_task_index(r, rounds_per_task, fcl_split.num_tasks)
            sampled = _sample_clients(r, fcl_split.num_clients, client_sample_frac, rng)
            logger.info(f"round={r} task={task_id} sampled_clients={sampled}")
            updates = []

            for cid in sampled:
                ds = fcl_split.client_dataset(cid, task_id=task_id)
                if len(ds) == 0:
                    continue
                if should_inject(backdoor, cid, r, total_rounds):
                    ds = PoisonedDataset(ds, backdoor)
                effective_ds = replay_buffer.effective_dataset(cid, ds) if replay_buffer is not None else ds
                client = Client(
                    client_id=cid,
                    dataset=effective_ds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    clip_norm=clip_norm,
                    device=device,
                    text_mode=text_mode,
                )
                model = model_factory().to(device)
                update = client.local_update(server.global_state, model, round_index=r)
                updates.append(update)
                if replay_buffer is not None:
                    replay_buffer.admit(cid, ds)

                if summarise_fn is not None and r % summary_frequency == 0:
                    payload = summarise_fn(
                        model=model, dataset=ds, device=device,
                        eps=summary_eps, delta=summary_delta,
                    )
                    summary = commit_summary(
                        client_id=cid, round_index=r,
                        summary_type=payload.get("type", "unknown"),
                        payload=payload, eps=summary_eps, delta=summary_delta,
                    )
                    server.store_summary(summary)

            server.apply_round(updates)

            if evaluate_fn is not None and (r + 1) % evaluate_every == 0:
                metrics = evaluate_fn(server, fcl_split, task_id=task_id)
                logger.info(f"round={r+1} task={task_id} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            # Numbered checkpoint (for archival) + resume snapshot (for resume).
            if checkpoint_dir is not None and (r + 1) % evaluate_every == 0:
                torch.save(
                    {
                        "round": r + 1,
                        "global_state": server.global_state,
                        "decay_state": server.decay_state.tilde_delta,
                        "decay_alpha": server.decay_state.alpha,
                        "summaries": server.summaries,
                    },
                    checkpoint_dir / f"ckpt_r{r+1}.pt",
                )
            if checkpoint_dir is not None and (r + 1) % checkpoint_every == 0:
                _flush_snapshot()

    # Final snapshot on clean exit.
    if checkpoint_dir is not None:
        _flush_snapshot()

    return server
