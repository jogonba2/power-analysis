from typing import Callable

import catalogue
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

    p_value = mcnemar(
        table=test_params.simulated_dataset.data,
        exact=test_params.exact,
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
    _, p_value = proportions_ztest(
        count=test_params.simulated_dataset.data,
        nobs=test_params.simulated_dataset.dataset_size,
        alternative="two-sided",
    )

    effect = effect_fn(sample=test_params.simulated_dataset)
    return StatsTestOutput(p_value=p_value, effect=effect)


@stats_tests.register("stats_test::paired_z")
def paired_ztest(
    args: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    raise NotImplementedError("TODO")
