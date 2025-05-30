import numpy as np

from .logging import get_logger
from .types import PowerOutput, StatsTestParameters

_logger = get_logger(__name__)


def compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    iterations: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
):
    assert iterations > 0, "Number of iterations must be greater than 0"
    true_effect = true_effect_fn()

    # If the true effect is 0, Null is true, so return NaNs
    if not true_effect.all():
        return PowerOutput(
            power=np.nan, mean_eff=np.nan, type_m=np.nan, type_s=np.nan
        )

    rng = np.random.default_rng(seed=seed)
    p_values, effects = [], []

    for _ in range(iterations):
        # simulate a dataset (sample) from the given DGP:
        simulated_sample = data_generating_fn(rng=rng)
        test_parameters = StatsTestParameters(
            simulated_sample=simulated_sample.data, exact=False
        )
        # run the hypothesis test:
        output = hypothesis_test_fn(test_parameters)
        p_values.append(output.p_value)
        effects.append(output.effect)

    p_values = np.array(p_values)
    effects = np.array(effects)

    true_sign = np.sign(true_effect) if not np.isnan(true_effect) else 0
    # filter the significant effects
    significant_effects = [
        effect for effect, pval in zip(effects, p_values) if pval <= alpha
    ]

    power = len(significant_effects) / iterations

    # mean effect across all effects (not just the significant ones)
    # TODO: is this what we want? Or do we neeed to average only the significant effects?
    mean_eff = np.mean(effects).item() if effects.any() else np.nan

    type_m = (
        np.mean(
            [abs(effect) / abs(true_effect) for effect in significant_effects]
        ).item()
        if significant_effects and true_effect != 0
        else np.nan
    )
    type_s = (
        np.mean(
            [np.sign(effect) != true_sign for effect in significant_effects]
        ).item()
        if significant_effects
        else np.nan
    )

    return PowerOutput(
        power=power, mean_eff=mean_eff, type_m=type_m, type_s=type_s
    )
