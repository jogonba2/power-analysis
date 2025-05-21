"""
The test here is done thinking on McNemar test, but it is general enough to be extended to other statistics.
"""

from functools import partial

import numpy as np
import pytest

from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters, PowerOutput, StatsTestParameters


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


def true_effect(true_prob_table: np.ndarray) -> float:
    return true_prob_table[0, 1] - true_prob_table[1, 0]


@pytest.fixture
def true_effect_fn(true_prob_table):
    return partial(true_effect, true_prob_table=true_prob_table)


@pytest.fixture
def seed():
    return 13


def compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    iterations,
    alpha,
    seed,
):
    rng = np.random.default_rng(seed=seed)
    p_values, effects = [], []
    for _ in range(iterations):
        dgp = data_generating_fn(rng=rng)
        test_parameters = StatsTestParameters(
            simulated_sample=dgp.data, exact=False
        )
        output = hypothesis_test_fn(test_parameters)
        p_values.append(output.p_value)
        effects.append(output.effect)
    p_values = np.array(p_values)
    effects = np.array(effects)

    true_effect = true_effect_fn()

    true_sign = np.sign(true_effect) if not np.isnan(true_effect) else 0
    sig = [(d, p) for d, p in zip(effects, p_values) if p <= alpha]

    power = (
        sum(1 for d, _ in sig if np.sign(d) == true_sign) / iterations
        if iterations > 0
        else np.nan
    )
    mean_eff = np.mean(effects) if effects.any() else np.nan
    type_m = (
        np.mean([abs(d) / abs(true_effect) for d, _ in sig])
        if sig and true_effect != 0
        else np.nan
    )
    type_s = (
        np.mean([np.sign(d) != true_sign for d, _ in sig]) if sig else np.nan
    )

    return PowerOutput(
        power=power, mean_eff=mean_eff, type_m=type_m, type_s=type_s
    )


@pytest.fixture
def iterations():
    return 11


@pytest.fixture
def alpha():
    return 0.05


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
