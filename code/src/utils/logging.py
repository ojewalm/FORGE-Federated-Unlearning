from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "fu", log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a named logger + also ensure the root "fu" logger is wired.

    federated_train, train_fedavg_and_maybe_store_history and the baseline
    modules all log via `logging.getLogger("fu")`, so any script that calls
    them must have the "fu" logger set up with handlers to see per-round
    output. If `name != "fu"`, we set up both the named logger and "fu",
    routing "fu" through the same handlers.
    """
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    def _configure(lg: logging.Logger, add_file: bool):
        if lg.handlers:
            return
        lg.setLevel(level)
        s = logging.StreamHandler(sys.stdout)
        s.setFormatter(formatter)
        lg.addHandler(s)
        if add_file and log_dir is not None:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(d / f"{lg.name}.log")
            fh.setFormatter(formatter)
            lg.addHandler(fh)

    named = logging.getLogger(name)
    _configure(named, add_file=True)
    if name != "fu":
        _configure(logging.getLogger("fu"), add_file=True)
    return named
