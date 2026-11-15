"""Resumable checkpointing: survives Ctrl-C, SIGTERM, crashes, sleep/wake.

Design:
- Every `checkpoint_every` rounds the loop calls `save_snapshot(state, path)`, which
  atomically writes a full pickle to `path.tmp` then renames it to `path`. Atomic
  rename guarantees that on a crash you either have the previous snapshot or the
  new one, never a half-written file.
- A SIGTERM / SIGINT handler flushes the last known state before exit so you lose
  at most `checkpoint_every - 1` rounds on a graceful kill.
- On startup the training script calls `resume_state(path)`; if the file exists
  it returns the state to seed the loop; otherwise it returns None and training
  starts fresh.
- All RNGs (Python, NumPy, Torch, Torch-CUDA) are captured so the same run
  produces the same output whether or not it was interrupted.
"""
from __future__ import annotations

import contextlib
import logging
import os
import pickle
import random
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


logger = logging.getLogger("fu")


@dataclass
class RngState:
    python: object
    numpy: dict
    torch_cpu: torch.Tensor
    torch_cuda_all: list

    @classmethod
    def capture(cls) -> "RngState":
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch_cpu=torch.random.get_rng_state(),
            torch_cuda_all=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.random.set_rng_state(self.torch_cpu)
        if torch.cuda.is_available() and self.torch_cuda_all:
            torch.cuda.set_rng_state_all(self.torch_cuda_all)


@dataclass
class ResumableSnapshot:
    """Full state needed to resume federated training exactly where it stopped."""
    round_index: int
    global_state: OrderedDict
    decay_tilde_delta: OrderedDict
    decay_alpha: float
    aggregator_gen_state: torch.Tensor      # DecayedAggregator's noise Generator
    summaries: dict                          # client_id -> CommittedSummary
    rng_state: RngState
    schema_version: int = 1
    saved_at: float = field(default_factory=time.time)
    extras: dict = field(default_factory=dict)   # dataset name, config hash, etc.


def save_snapshot(snapshot: ResumableSnapshot, path: Path) -> None:
    """Atomically write a snapshot. On crash the previous file is preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_snapshot(path: Path) -> ResumableSnapshot | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        snap = pickle.load(f)
    if not isinstance(snap, ResumableSnapshot):
        raise ValueError(f"File {path} is not a ResumableSnapshot")
    return snap


def resume_state(checkpoint_dir: Path) -> ResumableSnapshot | None:
    """Look for a `resume.pt` in the checkpoint dir and load it if present.

    Also considers `resume.pt.tmp` as a fallback if the atomic rename didn't
    complete (rare, but shows up on hard power-loss). Only the .tmp is loaded
    if the primary is missing, since the .tmp may be partially written.
    """
    primary = checkpoint_dir / "resume.pt"
    if primary.exists():
        try:
            return load_snapshot(primary)
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            logger.warning(f"Corrupted resume.pt at {primary}: {e}. Falling back to .tmp if present.")
    fallback = checkpoint_dir / "resume.pt.tmp"
    if fallback.exists():
        try:
            return load_snapshot(fallback)
        except Exception:
            return None
    return None


class SignalHandler:
    """Install SIGTERM/SIGINT handlers that trigger a final snapshot on exit.

    Usage:
        handler = SignalHandler(save_callback=lambda: save_snapshot(...))
        handler.install()
        ... training loop ...
        handler.uninstall()

    Or as a context manager:
        with SignalHandler(save_callback=...):
            ... training ...
    """

    def __init__(self, save_callback: Callable[[], None]):
        self.save_callback = save_callback
        self._prev_sigterm = None
        self._prev_sigint = None
        self._triggered = False

    def _handler(self, signum, frame):
        if self._triggered:
            return
        self._triggered = True
        logger.info(f"Received signal {signum}; flushing snapshot before exit ...")
        try:
            self.save_callback()
        except Exception as e:
            logger.error(f"Failed to save snapshot on signal: {e}")
        # Restore previous handler and re-raise the signal so the process actually exits.
        if signum == signal.SIGTERM and self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        elif signum == signal.SIGINT and self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
        os.kill(os.getpid(), signum)

    def install(self) -> None:
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._handler)
        self._prev_sigint = signal.signal(signal.SIGINT, self._handler)

    def uninstall(self) -> None:
        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.uninstall()


@contextlib.contextmanager
def resumable_run(checkpoint_dir: Path, save_callback: Callable[[], None]):
    """Combined context: install signal handlers, expose checkpoint dir."""
    with SignalHandler(save_callback=save_callback):
        yield checkpoint_dir
