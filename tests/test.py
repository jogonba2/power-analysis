"""
The test here is done thinking on McNemar test, but it is general enough to be extended to other statistics.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
import pytest
from numpy.random import Generator
from statsmodels.stats.contingency_tables import mcnemar

from power.types import PowerOutput


@dataclass
class DGPParameters:
    true_prob_table: np.ndarray
    dataset_size: int


@dataclass
class SimulatedSample:
    data: np.ndarray


@dataclass
class StatsTestParameters:
    """
    Base class for parameters for the test statistic function.
    """

    simulated_sample: SimulatedSample
    exact: bool

    @property
    def dataset_size(self):
        return self.simulated_sample.data.shape


@dataclass
class StatsTestOutput:
    p_value: float
    diff: float


@pytest.fixture
def expected_output() -> PowerOutput:
    return PowerOutput(
        power=1.0,
        mean_eff=0.20053000000000004,
        type_m=1.00265,
        type_s=0.0,
    )


def mcnemar_test(args: StatsTestParameters) -> StatsTestOutput:
    p_value = mcnemar(table=args.simulated_sample, exact=args.exact).pvalue
    # This is one of many ways of doing this. Now we are doing (a-b)/N
    # We have to think how to parameterize this because it could be another fn at some point
    diff = (
        args.simulated_sample[0, 1] - args.simulated_sample[1, 0]
    ) / args.dataset_size
    return StatsTestOutput(p_value=p_value, diff=diff)


@pytest.fixture
def hypothesis_test_fn():
    return mcnemar_test


def contingency_table(
    dgp_args: DGPParameters, rng: Generator
) -> SimulatedSample:
    data = rng.multinomial(
        dgp_args.dataset_size, dgp_args.true_prob_table.flatten()
    ).reshape(2, 2)
    return SimulatedSample(data=data)


@pytest.fixture
def true_prob_table():
    return np.array([[0.0, 0.5], [0.3, 0.2]])


@pytest.fixture
def data_generating_fn(true_prob_table):
    dgp_args = DGPParameters(true_prob_table=true_prob_table, dataset_size=1000)
    return partial(contingency_table, dgp_args=dgp_args)


def effect(true_prob_table: np.ndarray) -> float:
    return true_prob_table[0, 1] - true_prob_table[1, 0]


@pytest.fixture
def true_effect_fn():
    return effect


@pytest.fixture
def seed():
    return 13


def compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    repetitions,
    alpha,
    seed,
):
    rng = np.random.default_rng(seed=seed)
    p_values, diffs = [], []
    for _ in range(repetitions):
        dgp = data_generating_fn(rng=rng)
        test_parameters = StatsTestParameters(
            simulated_sample=dgp.data, exact=False
        )
        output = hypothesis_test_fn(test_parameters)
        p_values.append(output.p_value)
        diffs.append(output.diff)
    p_values = np.array(p_values)
    diffs = np.array(diffs)

    true_diff = true_effect_fn(...)
    true_sign = np.sign(true_diff) if not np.isnan(true_diff) else 0
    sig = [(d, p) for d, p in zip(diffs, p_values) if p <= alpha]
    power = (
        sum(1 for d, _ in sig if np.sign(d) == true_sign) / repetitions
        if repetitions > 0
        else np.nan
    )
    mean_eff = np.mean(diffs) if diffs else np.nan
    type_m = (
        np.mean([abs(d) / abs(true_diff) for d, _ in sig])
        if sig and true_diff != 0
        else np.nan
    )
    type_s = (
        np.mean([np.sign(d) != true_sign for d, _ in sig]) if sig else np.nan
    )

    return PowerOutput(
        power=power, mean_eff=mean_eff, type_m=type_m, type_s=type_s
    )


@pytest.fixture
def repetitions():
    return 500


@pytest.fixture
def alpha():
    return 0.05


def test_compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    repetitions,
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
        repetitions,
        alpha,
        seed,
    )
    assert expected_output == output
