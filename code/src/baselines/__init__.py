from .common import BaselineResult, train_fedavg_and_maybe_store_history
from .halimi import halimi_unlearn
from .fedmemprune import fedmemprune_unlearn
from .fedrecovery import fedrecovery_unlearn
from .federaser import federaser_unlearn
from .sifu import sifu_unlearn, sifu_choose_rollback

BASELINES = {
    "halimi": halimi_unlearn,
    "fedmemprune": fedmemprune_unlearn,
    "fedrecovery": fedrecovery_unlearn,
    "federaser": federaser_unlearn,
    "sifu": sifu_unlearn,
}


__all__ = [
    "BaselineResult",
    "train_fedavg_and_maybe_store_history",
    "halimi_unlearn",
    "fedmemprune_unlearn",
    "fedrecovery_unlearn",
    "federaser_unlearn",
    "sifu_unlearn",
    "sifu_choose_rollback",
    "BASELINES",
]

_MODULE_DOC = """Reimplementations of the five federated unlearning baselines we benchmark
against.

We reimplement rather than vendor for three reasons:

1. Two of the five reference repos (Accenture SIFU / FedEraser mirror) do not
   ship a licence file, so we cannot legally vendor.
2. FedRecovery has no public code at all.
3. Even for the permissively-licensed ones (IBM Halimi = Apache-2.0, JWei1999
   FedMemPrune = MIT), we need to feed each method the same base FedAvg
   checkpoint, the same evaluation pipeline, and the same M-series-friendly
   device abstraction. A uniform reimplementation is the cleanest way to make
   the comparison apples-to-apples.

Each baseline module is header-commented with (a) the paper citation, (b) the
reference repository, and (c) the licence of that reference. Where we
cross-check against the authors' code we say so explicitly. See
`ATTRIBUTION.md` in this directory for a consolidated pointer list.
"""

__doc__ = _MODULE_DOC
