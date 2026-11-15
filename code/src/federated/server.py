"""Server: holds global model, decay state, and pre-committed summaries."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .aggregator import DecayState, DecayedAggregator
from .client import ClientUpdate
from .params import ParamDict


@dataclass
class CommittedSummary:
    """A DP summary from a client, cryptographically bound to their id."""
    client_id: int
    round_index: int
    summary_type: str
    payload: Any                 # opaque; interpretation determined by summary_type
    eps: float
    delta: float
    commitment: str              # hex digest of payload + client_id + round_index


def _hash_payload(client_id: int, round_index: int, payload_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(client_id.to_bytes(8, "little"))
    h.update(round_index.to_bytes(8, "little"))
    h.update(payload_bytes)
    return h.hexdigest()


def commit_summary(
    client_id: int,
    round_index: int,
    summary_type: str,
    payload: Any,
    eps: float,
    delta: float,
) -> CommittedSummary:
    """Build a CommittedSummary. Payload can be any pickleable object."""
    import pickle
    payload_bytes = pickle.dumps(payload)
    commitment = _hash_payload(client_id, round_index, payload_bytes)
    return CommittedSummary(
        client_id=client_id,
        round_index=round_index,
        summary_type=summary_type,
        payload=payload,
        eps=eps,
        delta=delta,
        commitment=commitment,
    )


@dataclass
class Server:
    """Federated server state."""
    global_state: ParamDict
    decay_state: DecayState
    aggregator: DecayedAggregator
    round_index: int = 0
    summaries: dict[int, CommittedSummary] = field(default_factory=dict)

    def apply_round(self, updates: list[ClientUpdate]) -> None:
        new_global, new_decay = self.aggregator.aggregate(self.global_state, updates, self.decay_state)
        self.global_state = new_global
        self.decay_state = new_decay
        self.round_index += 1

    def store_summary(self, summary: CommittedSummary) -> None:
        """Keep only the latest summary per client (bounded storage)."""
        self.summaries[summary.client_id] = summary

    def get_summary(self, client_id: int) -> CommittedSummary | None:
        return self.summaries.get(client_id)

    def purge_summary(self, client_id: int) -> None:
        self.summaries.pop(client_id, None)

    def load_into(self, model: nn.Module) -> nn.Module:
        model.load_state_dict(self.global_state)
        return model
