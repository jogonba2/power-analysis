import numpy as np

from .logging import get_logger
from .types import PowerOutput, StatsTestParameters

_logger = get_logger(__name__)


def compute_power(
    data_generating_fn,
    hypothesis_test_fn,
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

    # TODO: Multiprocessing in the for loop
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


# def power_bounds(
#     baseline_acc: float,
#     delta_acc: float,
#     dataset_size: int,
#     alpha: float = 0.05,
#     r: int = 5000,
#     test_type: str = "mcnemar",
# ) -> PowerBounds:
#     acc1, acc2 = baseline_acc, baseline_acc + delta_acc
#     # Upper‐bound (max agreement)
#     p_both_corr = min(acc1, acc2)
#     p_diff = abs(acc1 - acc2)
#     p_both_inc = 1 - max(acc1, acc2)
#     if acc2 > acc1:
#         pu_tab = np.array(
#             [[p_both_inc, 0.0], [p_diff, p_both_corr]], dtype="float32"
#         )
#     else:
#         pu_tab = np.array(
#             [[p_both_inc, p_diff], [0.0, p_both_corr]], dtype="float32"
#         )
#     upper_bound = compute_power(pu_tab, dataset_size, alpha, r, test_type)
#
#     # Lower‐bound (max disagreement)
#     if (2 - acc1 - acc2) <= 1:
#         p_neither = 0.0
#         only1 = 1 - acc1
#         only2 = 1 - acc2
#         p_both = 1 - only1 - only2
#     else:
#         p_both = 0.0
#         only1 = acc1
#         only2 = acc2
#         p_neither = 1 - only1 - only2
#     pl_tab = np.array([[p_neither, only1], [only2, p_both]], dtype="float32")
#     lower_bound = compute_power(pl_tab, dataset_size, alpha, r, test_type)
#
#     return PowerBounds(upper=upper_bound, lower=lower_bound)
