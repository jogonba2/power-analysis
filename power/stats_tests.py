from typing import Callable

import catalogue
from statsmodels.stats.contingency_tables import mcnemar

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
    effect = effect_fn(args.simulated_sample, args.dataset_size)
    return StatsTestOutput(p_value=p_value, effect=effect)
