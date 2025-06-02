import numpy as np

from .logging import get_logger
from .types import PowerOutput, StatsTestParameters

_logger = get_logger(__name__)


def compute_power(
    data_generating_fn,
    hypothesis_test_fn,
    true_effect_fn,
    iterations: int = 1_000,
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

    p_values = np.empty(iterations, dtype=float)
    effects = np.empty(iterations, dtype=float)

    for i in range(iterations):
        simulated_dataset = data_generating_fn(rng=rng)
        out = hypothesis_test_fn(
            test_params=StatsTestParameters(
                simulated_dataset=simulated_dataset, exact=False
            )
        )
        p_values[i] = out.p_value
        effects[i] = out.effect

    true_sign = np.sign(true_effect) if not np.isnan(true_effect) else 0
    # filter the significant effects
    significance_mask = p_values <= alpha

    significant_effects = effects[significance_mask]

    power = len(significant_effects) / len(effects)

    # mean effect across all effects (not just the significant ones)
    # TODO: is this what we want? Or do we neeed to average only the significant effects?
    mean_eff = np.mean(effects).item() if effects.any() else np.nan

    type_m = (
        np.mean(
            [abs(effect) / abs(true_effect) for effect in significant_effects]
        ).item()
        if significant_effects.any() and true_effect != 0
        else np.nan
    )
    type_s = (
        np.mean(
            [np.sign(effect) != true_sign for effect in significant_effects]
        ).item()
        if significant_effects.any()
        else np.nan
    )

    return PowerOutput(
        power=power, mean_eff=mean_eff, type_m=type_m, type_s=type_s
    )
