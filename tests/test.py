"""
The test here is done thinking on McNemar test, but it is general enough to be extended to other statistics.
"""

from functools import partial

import numpy as np
import pytest

from power.compute_power import compute_power
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters, PowerOutput


@pytest.fixture
def expected_output() -> PowerOutput:
    return PowerOutput(
        power=1.0,
        mean_eff=np.float64(0.19336363636363635),
        type_m=np.float64(0.9668182394450396),
        type_s=np.float64(0.0),
    )


@pytest.fixture
def effect_test_fn():
    return effects.get("effect::cohens_g")


@pytest.fixture
def hypothesis_test_fn(effect_test_fn):
    return partial(
        stats_tests.get("stats_test::mcnemar"), effect_fn=effect_test_fn
    )


@pytest.fixture
def true_prob_table():
    return np.array([[0.0, 0.5], [0.3, 0.2]], dtype="float32")


@pytest.fixture
def data_generating_fn(true_prob_table):
    dgp_args = DGPParameters(true_prob_table=true_prob_table, dataset_size=1000)
    return partial(dgps.get("dgp::contingency_table"), dgp_args=dgp_args)


@pytest.fixture
def true_effect_fn(true_prob_table):
    return partial(
        effects.get("effect::cohens_g"),
        sample=true_prob_table,
        true_effect=True,
        dataset_size=sum(true_prob_table),
    )


@pytest.mark.parametrize(["seed", "iterations", "alpha"], [(13, 11, 0.05)])
def test_compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    iterations,
    alpha,
    seed,
    expected_output,
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
    assert expected_output == output
