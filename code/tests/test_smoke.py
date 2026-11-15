"""Smoke tests: check that the main modules import and the DP math is sane."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    from src.utils.dp import forward_forgetting_epsilon, cooling_off_time, composed_budget
    from src.utils import load_config
    from src.federated import DecayedAggregator, Server
    from src.datasets.partition import dirichlet_partition, task_partition_by_class
    from src.models import build_model
    from src.unlearning import SUMMARISERS, non_interactive_erasure
    from src.attacks import mia_yeom, mia_lira_lite
    assert set(SUMMARISERS.keys()) == {"moments", "logits", "synthetic", "sketch"}


def test_forward_forgetting_monotone_in_T():
    from src.utils.dp import forward_forgetting_epsilon
    eps_10 = forward_forgetting_epsilon(T=10, alpha=0.9, clip_C=1.0, sigma=0.5, rho=0.1, delta=1e-5)
    eps_100 = forward_forgetting_epsilon(T=100, alpha=0.9, clip_C=1.0, sigma=0.5, rho=0.1, delta=1e-5)
    assert eps_100 < eps_10, f"Expected monotone decrease: eps(10)={eps_10}, eps(100)={eps_100}"
    assert eps_100 > 0


def test_cooling_off_time():
    from src.utils.dp import cooling_off_time
    T = cooling_off_time(eps_target=0.5, alpha=0.9, clip_C=1.0, sigma=0.5, rho=0.1, delta=1e-5)
    assert 0 < T < 500, f"Cooling-off time out of expected range: {T}"


def test_composition():
    from src.utils.dp import composed_budget
    e_seq, d_seq = composed_budget(0.3, 0.4, 1e-5, 1e-5, parallel=False)
    e_par, d_par = composed_budget(0.3, 0.4, 1e-5, 1e-5, parallel=True)
    assert math.isclose(e_seq, 0.7)
    assert math.isclose(e_par, 0.4)


def test_dirichlet_partition_balanced_alpha_large():
    import numpy as np
    from src.datasets.partition import dirichlet_partition
    labels = np.array([i % 10 for i in range(1000)])
    parts = dirichlet_partition(labels, num_clients=10, alpha=100.0, seed=0)
    sizes = [len(p) for p in parts]
    # With large alpha, each client should have roughly similar size.
    assert max(sizes) - min(sizes) < 200
    assert sum(sizes) == len(labels)


if __name__ == "__main__":
    test_imports()
    test_forward_forgetting_monotone_in_T()
    test_cooling_off_time()
    test_composition()
    test_dirichlet_partition_balanced_alpha_large()
    print("All smoke tests passed.")
