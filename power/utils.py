from functools import partial

import numpy as np

from power.types import PowerOutput

from .compute_power import compute_power
from .dgps import dgps
from .effects import effects
from .stats_tests import stats_tests
from .types import DGPParameters


def make_probability_table(
    baseline_acc: float = 0.5,  # accuracy of the baseline model (Model 1/A)
    delta_acc: float = 0.1,  # accuracy difference between Model 1 and Model 2, \theta_B - baseline_acc
    agreement_rate: float = 0.8,  # agreement rate between the two models (Model 1 and Model 2)
) -> np.ndarray:
    """
    Returns a 2x2 probability table representing the joint distribution of two models predictions.
    The table is structured as follows:
        [
            [p_both_incorrect, p_only_2_correct],
            [p_only_1_correct, p_both_correct]
        ]

    where:
      - the baseline model is model 1, so that accuracy_model1 = table[1, :].sum() = baseline_acc
      - the second model, model 2, has accuracy_model2 = table[:, 1].sum() = baseline_acc + delta_acc
      - the agreement rate is the sum of the probabilities of both models giving the same prediction, i.e.,
            table.diagonal.sum() = agreement_rate
    """
    # p11 is the probability that both models are correct
    p11 = (agreement_rate + 2 * baseline_acc + delta_acc - 1) / 2
    p10 = baseline_acc - p11
    p01 = baseline_acc + delta_acc - p11
    p00 = agreement_rate - p11
    probs = np.array([p00, p01, p10, p11])
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError(
            "The provided parameters do not yield a valid probability table."
        )

    table = probs.reshape(2, 2)
    # some more sanity checks
    assert np.isclose(table[1, :].sum(), baseline_acc), "Error!"
    assert np.isclose(table[:, 1].sum(), baseline_acc + delta_acc), "Error!"
    assert np.isclose(table.diagonal().sum(), agreement_rate), "Error!"

    return table


def estimate_power_from_accuracy(
    baseline_acc: float,
    delta_acc: float,
    dataset_size: int,
    agreement: float = 0.75,
    alpha: float = 0.05,
    iterations: int = 1000,
    seed: int = 42,
    dgp: str = "dgp::contingency_table",
    test: str = "stats_test::mcnemar",
    effect: str = "effect::cohens_g",
) -> PowerOutput:
    """
    Estimate statistical power given baseline and delta accuracy.

    Returns:
        pd.Series with power, mean_effect, type_m, type_s
    """
    try:
        prob_table = make_probability_table(baseline_acc, delta_acc, agreement)
    except ValueError:
        return PowerOutput(
            power=np.nan, mean_eff=np.nan, type_m=np.nan, type_s=np.nan
        )

    dgp_params = DGPParameters(
        true_prob_table=prob_table, dataset_size=dataset_size
    )

    data_generating_fn = partial(dgps.get(dgp), dgp_params=dgp_params)
    effect_fn = effects.get(effect)
    statistical_test_fn = partial(stats_tests.get(test), effect_fn=effect_fn)
    true_effect_fn = partial(effect_fn, dgp_params=dgp_params)

    return compute_power(
        data_generating_fn=data_generating_fn,
        hypothesis_test_fn=statistical_test_fn,
        true_effect_fn=true_effect_fn,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )
