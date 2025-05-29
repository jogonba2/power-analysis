"""
This test compare the results of the simulation of the unpaired z-test
against NormalIndPower's closed form solution; basically, it's a sanity
check for the simulation code.
"""

from functools import partial

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from power.compute_power import compute_power
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters


@pytest.fixture
def effect_test_fn():
    return effects.get("effect::risk_difference")


@pytest.fixture
def hypothesis_test_fn(effect_test_fn):
    return partial(
        stats_tests.get("stats_test::unpaired_z"), effect_fn=effect_test_fn
    )


@pytest.fixture
def true_prob_table():
    return np.array([[0.0, 0.5], [0.3, 0.2]], dtype="float32")


@pytest.fixture
def data_generating_fn(true_prob_table):
    dgp_args = DGPParameters(true_prob_table=true_prob_table, dataset_size=47)
    return partial(dgps.get("dgp::contingency_table"), dgp_args=dgp_args)


@pytest.fixture
def true_effect_fn(true_prob_table):
    return partial(
        effects.get("effect::risk_difference"),
        true_prob_table=true_prob_table,
        sample=None,
        dataset_size=47,
    )


@pytest.mark.parametrize(["seed", "iterations", "alpha"], [(42, 5000, 0.05)])
def test_compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    iterations,
    alpha,
    seed,
):
    """
    ...
    Args:
        data_generating_fn (): returns a contingency table for classification ...
        hypothesis_test_fn (Callable[[np.ndarray], float]): mcnemar test ... picks a simulated sample and returns a p value
        true_effect_fn (...):
    """
    output = compute_power(
        data_generating_fn,
        hypothesis_test_fn,
        true_effect_fn,
        iterations,
        alpha,
        seed,
    )

    np.testing.assert_almost_equal(
        actual=output.power,
        desired=NormalIndPower(ddof=0).power(
            effect_size=proportion_effectsize(0.5, 0.3),
            nobs1=47,
            alpha=0.05,
            ratio=1,
            alternative="two-sided",
        ),
        decimal=1,
    )
