# tests/test_compute_power_speed.py
"""
Timing harness for `compute_power`.

    pytest -q --benchmark-save=baseline          # first run
    pytest -q --benchmark-compare                # later runs
"""

import numpy as np
import pytest

from power.compute_power import compute_power
from power.types import PowerOutput

from .test import data_generating_fn  # or wherever they live


@pytest.mark.parametrize("iterations", [5, 10, 11])
def test_compute_power_benchmark(
    benchmark,
    iterations,
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    alpha,
    seed,
):
    """
    Benchmark `compute_power` at several Monte-Carlo sizes.

    Parameters
    ----------
    iterations : int [parametrized]
        Number of simulated replications in `compute_power`.
    benchmark : pytest-benchmark fixture
        Times the wrapped call and records stats for later comparison.
    """

    result: PowerOutput = benchmark(
        compute_power,
        data_generating_fn,
        hypothesis_test_fn,
        true_effect_fn,
        iterations,
        alpha,
        seed,
    )

    # Sanity: the function still returns a PowerOutput
    assert isinstance(result, PowerOutput)
    # (Optional) quick smoke-check the fields are finite numbers
    assert np.isfinite(result.power)
