"""Cost metrics: server-side storage and per-round communication."""
from __future__ import annotations

import pickle
from typing import Any


def storage_footprint(payload: Any) -> int:
    """Return the pickled byte-size of a payload. Handy for measuring what the
    server retains (summaries, decay state) vs baselines (per-round updates)."""
    return len(pickle.dumps(payload))
