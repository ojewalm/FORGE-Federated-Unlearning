"""Parameter (state_dict) utilities: flatten, unflatten, arithmetic, norms."""
from __future__ import annotations

from collections import OrderedDict

import torch


ParamDict = "OrderedDict[str, torch.Tensor]"


def sub_state(a: ParamDict, b: ParamDict) -> ParamDict:
    return OrderedDict((k, a[k].detach() - b[k].detach()) for k in a)


def add_state(a: ParamDict, b: ParamDict, scale: float = 1.0) -> ParamDict:
    return OrderedDict((k, a[k].detach() + scale * b[k].detach()) for k in a)


def scale_state(a: ParamDict, s: float) -> ParamDict:
    return OrderedDict((k, s * v.detach()) for k, v in a.items())


def zero_state_like(a: ParamDict) -> ParamDict:
    return OrderedDict((k, torch.zeros_like(v)) for k, v in a.items())


def state_norm(a: ParamDict) -> float:
    return float(torch.sqrt(sum((v.detach() ** 2).sum() for v in a.values())))


def clip_state(a: ParamDict, max_norm: float) -> ParamDict:
    n = state_norm(a)
    if n <= max_norm or n < 1e-12:
        return a
    scale = max_norm / n
    return scale_state(a, scale)


def add_gaussian_noise(a: ParamDict, sigma: float, device=None) -> ParamDict:
    out = OrderedDict()
    for k, v in a.items():
        noise = torch.randn_like(v) * sigma
        out[k] = v + noise
    return out


def state_dict_by_prefix(state: ParamDict, prefix: str) -> ParamDict:
    return OrderedDict((k, v) for k, v in state.items() if k.startswith(prefix))


def merge_states(base: ParamDict, overlay: ParamDict) -> ParamDict:
    """Return base with overlay's keys overwritten."""
    merged = OrderedDict(base)
    for k, v in overlay.items():
        merged[k] = v
    return merged
