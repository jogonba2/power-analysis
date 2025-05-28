import numpy as np

from .logging import get_logger
from .types import PowerOutput, StatsTestParameters

_logger = get_logger(__name__)


def compute_power(
    data_generating_fn,
    statistical_test_fn,
    true_effect_fn,
    iterations,
    alpha,
    seed,
):
    true_effect = true_effect_fn()

    # If the true effect is 0, Null is true, so return NaNs
    if true_effect == 0.0:
        return PowerOutput(
            power=np.nan, mean_eff=np.nan, type_m=np.nan, type_s=np.nan
        )

    rng = np.random.default_rng(seed=seed)
    p_values, effects = [], []

    for _ in range(iterations):
        dgp = data_generating_fn(rng=rng)
        test_parameters = StatsTestParameters(
            simulated_sample=dgp.data, exact=False
        )
        output = statistical_test_fn(test_parameters)
        p_values.append(output.p_value)
        effects.append(output.effect)
    p_values = np.array(p_values)
    effects = np.array(effects)

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
