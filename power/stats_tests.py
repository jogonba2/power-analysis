import math
from typing import Callable

import catalogue
import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t  # type: ignore
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest

from .types import StatsTestOutput, StatsTestParameters

stats_tests = catalogue.create("power", "stats_tests")


@stats_tests.register("stats_test::mcnemar")
def mcnemar_test(
    test_params: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Runs the statistical test and returns p value and effect.
    """
    data = test_params.simulated_dataset.data
    # check if b + c is 0, if so, we cannot run the test
    if data[0, 1] + data[1, 0] == 0:
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))
    p_value = mcnemar(
        table=data, exact=test_params.exact, correction=False
    ).pvalue  # type: ignore
    # This is one of many ways of doing this. Now we are doing (a-b)/N
    # We have to think how to parameterize this because it could be another fn at some point
    effect = effect_fn(sample=test_params.simulated_dataset)
    return StatsTestOutput(p_value=p_value, effect=effect)


@stats_tests.register("stats_test::unpaired_z")
def unpaired_ztest(
    test_params: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Runs the statistical test and returns p value and effect.
    """
    # zstat, pvalue
    count = test_params.simulated_dataset.data
    if (
        count.sum() == 0
        or count.sum() == test_params.simulated_dataset.dataset_size.sum()
    ):
        # If there are no successes or all successes, the std would be 0 and dont run the test
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))
    _, p_value = proportions_ztest(
        count=test_params.simulated_dataset.data,
        nobs=test_params.simulated_dataset.dataset_size,
        alternative="two-sided",
    )

    effect = effect_fn(sample=test_params.simulated_dataset)
    return StatsTestOutput(p_value=p_value, effect=effect)


@stats_tests.register("stats_test::paired_z")
def paired_ztest(
    test_params: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Paired z-test for two models evaluated on the same items (binary outcome).
    """
    data = test_params.simulated_dataset.data
    b = data[0, 1]  # A wrong, B right
    c = data[1, 0]  # A right, B wrong

    # If there are no disagreements, cannot compute
    if b + c == 0:
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))

    # z statistic
    z_stat = (b - c) / np.sqrt(b + c)

    # Two-sided p-value from standard normal
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    effect = effect_fn(sample=test_params.simulated_dataset)
    return StatsTestOutput(p_value=p_value, effect=effect)


@stats_tests.register("stats_test::paired_t")
def paired_t(
    test_params: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Paired t-test computed from a 2x2 paired contingency table:
        [[a, b],
         [c, d]]
    """

    data = np.asarray(test_params.simulated_dataset.data, dtype=float)
    if data.shape != (2, 2):
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))

    a, b = data[0, 0], data[0, 1]  # A−,B− ; A−,B+
    c, d = data[1, 0], data[1, 1]  # A+,B− ; A+,B+
    N = a + b + c + d
    if N < 2:
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))

    dbar = (c - b) / N
    sumsq = b + c  # (-1)^2*b + 0^2*(a+d) + (+1)^2*c
    s2_num = sumsq - N * (dbar**2)
    if s2_num <= 0:
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))
    s = math.sqrt(s2_num / (N - 1))

    if s == 0:
        return StatsTestOutput(p_value=float("nan"), effect=float("nan"))

    t_stat = dbar / (s / math.sqrt(N))
    df = int(N - 1)

    # Two-sided p-value
    try:
        p_value = 2.0 * (1.0 - student_t.cdf(abs(t_stat), df))
    except Exception:
        # Fallback: normal approximation
        p_value = 2.0 * (
            1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2)))
        )

    effect = effect_fn(sample=test_params.simulated_dataset)
    return StatsTestOutput(p_value=float(p_value), effect=effect)
