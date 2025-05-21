from typing import Callable

import catalogue
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest

from .types import StatsTestOutput, StatsTestParameters

stats_tests = catalogue.create("power", "stats_tests")


@stats_tests.register("stats_test::mcnemar")
def mcnemar_test(
    args: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Runs the statistical test and returns p value and effect.
    """
    p_value = mcnemar(table=args.simulated_sample, exact=args.exact).pvalue
    # This is one of many ways of doing this. Now we are doing (a-b)/N
    # We have to think how to parameterize this because it could be another fn at some point
    effect = effect_fn(args.simulated_sample, args.dataset_size, False)
    return StatsTestOutput(p_value=p_value, effect=effect)


@stats_tests.register("stats_test::unpaired_z")
def unpaired_ztest(
    args: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    """
    Runs the statistical test and returns p value and effect.
    """

    count = [
        args.simulated_sample.data[:, 1].sum(),
        args.simulated_sample.data[1:, :].sum(),
    ]

    nobs = [args.dataset_size, args.dataset_size]

    p_value = proportions_ztest(count=count, nobs=nobs)

    # TODO effect=...

    return StatsTestOutput(p_value=p_value, effect=0.0)


@stats_tests.register("stats_test::paired_z")
def paired_ztest(
    args: StatsTestParameters, effect_fn: Callable
) -> StatsTestOutput:
    raise NotImplementedError("TODO")
