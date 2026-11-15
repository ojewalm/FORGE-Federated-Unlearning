"""Device selection helper with M-series MPS support.

`resolve_device("auto")` picks:
  - cuda if available (NVIDIA)
  - mps if available and stable (Apple Silicon, macOS >= 12.3, torch >= 1.12)
  - cpu otherwise

Passing an explicit device string ("cpu", "cuda", "mps") is respected. If MPS
is requested but unavailable, we fall back to CPU with a warning rather than
erroring — long M4 runs shouldn't crash because someone toggled a system
setting.
"""
from __future__ import annotations

import logging
import os

import torch


logger = logging.getLogger("fu")


def _mps_available() -> bool:
    if not hasattr(torch.backends, "mps"):
        return False
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def resolve_device(request: str) -> str:
    req = (request or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("cuda requested but unavailable; falling back to cpu")
        return "cpu"
    if req == "mps":
        if _mps_available():
            # PyTorch's MPS backend still has partial fp64 support and some
            # ops fall back to CPU. Setting PYTORCH_ENABLE_MPS_FALLBACK=1
            # tells torch to CPU-fallback silently instead of erroring.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        logger.warning("mps requested but unavailable; falling back to cpu")
        return "cpu"
    return req  # "cpu" or anything else the caller wants
