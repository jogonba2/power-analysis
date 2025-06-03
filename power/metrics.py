from functools import partial
from typing import Optional

import numpy as np
from scipy import stats

from power.types import PowerOutput

from .compute_power import compute_power
from .dgps import dgps
from .effects import effects
from .stats_tests import stats_tests
from .types import DGPParameters, PowerOutput


def estimate_power_from_predictions(
    preds_a: list | np.ndarray,
    preds_b: list | np.ndarray,
    references: list | np.ndarray,
    dgp_fn: str = "contingency_table",
    statistical_test_fn: str = "mcnemar",
    effect_fn: str = "cohens_g",
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> PowerOutput:
    """
    Wrapper to compute power in classification scenarios.

    This function compares two sets of classification predictions against ground truth labels
    using a specified data-generating process (DGP), statistical test, and effect size function.
    It performs statistical analysis via simulation to estimate the power of detecting a significant
    difference between the classifiers.

    Args:
        preds_a (list | np.ndarray): Predictions from classifier A.
        preds_b (list | np.ndarray): Predictions from classifier B.
        references (list | np.ndarray): Ground truth labels.
        dgp_fn (str): The name of the data-generating process function to use for simulation.
                      Defaults to "contingency_table".
        statistical_test_fn (str): The name of the statistical test function to apply.
                                   Defaults to "mcnemar".
        effect_fn (str): The name of the effect size function to use.
                         Defaults to "cohens_g".
        iterations (int): Number of iterations to run for the simulation.
                          Defaults to 1000.
        alpha (float): Significance level to determine statistical power.
                       Defaults to 0.05.
        seed (int): Random seed for reproducibility.
                    Defaults to 13.

    Returns:
        PowerOutput: an object containing the statistical power, type S and M errors, and mean effect size.
    """
    # Create arrays for predictions and references
    if isinstance(preds_a, list):
        preds_a = np.array(preds_a)
    if isinstance(preds_b, list):
        preds_b = np.array(preds_b)
    if isinstance(references, list):
        references = np.array(references)

    # Create the true probability table
    dataset_size = len(references)
    true_prob_table = np.array(
        [
            [
                ((preds_a != references) & (preds_b != references)).mean(),
                ((preds_a == references) & (preds_b != references)).mean(),
            ],
            [
                ((preds_a != references) & (preds_b == references)).mean(),
                ((preds_a == references) & (preds_b == references)).mean(),
            ],
        ]
    )

    # Instantiate the DGP
    dgp_args = DGPParameters(
        true_prob_table=true_prob_table, dataset_size=dataset_size
    )

    data_generating_fn = partial(dgps.get(f"dgp::{dgp_fn}"), dgp_args=dgp_args)

    # Instantiate statistical test and effect estimation
    effect_fn = effects.get(f"effect::{effect_fn}")

    statistical_test_fn = partial(
        stats_tests.get(f"stats_test::{statistical_test_fn}"),
        effect_fn=effect_fn,
    )

    # Prepare the function to compute the *true* effect size for benchmarking
    true_effect_fn = partial(
        effect_fn,
        true_prob_table=true_prob_table,
        sample=None,
        dataset_size=dataset_size,
    )

    # Compute the power
    return compute_power(
        data_generating_fn,
        statistical_test_fn,
        true_effect_fn,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )


def find_dataset_size(
    mde: float,
    alpha: float = 0.05,
    beta: float = 0.2,
    k_a: int = 1,
    k_b: int = 1,
    x_a: Optional[list | np.ndarray] = None,
    x_b: Optional[list | np.ndarray] = None,
    var_a: Optional[float] = None,
    var_b: Optional[float] = None,
    omega: Optional[float] = None,
) -> int:
    """
    Find the dataset size for a fixed Minimum Detectable Effect (MDE)
    with a Type I error rate (alpha) and a Type II error rate (beta).

    See Section 5. of https://arxiv.org/pdf/2411.00640 for more specific details.

    This function can estimate the dataset size in two setups:

    1. From data: you can provide `x_a` and `x_b` to estimate the dataset size required to find the provided MDE.
                  Both `x_a` and `x_b` must be of shape (N,), containing previous evaluation data of models A and B.

    2. From fixed parameters: you can fix `var_a`, `var_b` and `omega` at some sensible level. For instance,
                              `var_a=0`, `var_b=0`, and `omega=1/9` as shown in https://arxiv.org/pdf/2411.00640.

    Therefore, do not provide `x_a` and `x_b` when providing `var_a`, `var_b`, and `omega` (and viceversa).

    Args:
        mde (float): minimum detectable effect you want to detect.
        alpha (float): type I error rate. Default to 0.05.
        beta (float): type II error rate. Default to 0.2.
        k_a (int): number of answers from model A that will be sampled in a paired analysis. Default to 1.
        k_b (int): number of answers from model B that will be sampled in a paired analysis. Default to 1.
        x_a (Optional[list | np.ndarray]): previous evaluation data of model A.
        x_b (Optional[list | np.ndarray]): previous evaluation data of model B.
        var_a (Optional[float]): variance of model A accuracy.
        var_b (Optional[float]): variance of model B accuracy.
        omega (Optional[float]): defined as `var_a` - `var_b` - 2Cov(`x_a`, `x_b`)

    Returns:
        int: dataset size required to detect a MDE.
    """

    from_data = [x_a, x_b]
    from_params = [var_a, var_b, omega]

    data_provided = all(x is not None for x in from_data)
    params_provided = all(x is not None for x in from_params)

    assert (data_provided and not params_provided) or (
        params_provided and not data_provided
    ), "You must provide either (x_a, x_b) or (var_a, var_b, omega), not both or any."

    if x_a is not None and x_b is not None:
        var_a = np.var(x_a)
        var_b = np.var(x_b)
        cov_ab = np.cov(x_a, x_b)[0, 1]
        omega = var_a + var_b - 2 * cov_ab

    z_alpha = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)

    num = ((z_alpha + z_beta) ** 2) * (omega + (var_a / k_a) + (var_b / k_b))

    return num // (mde**2)


def find_minimum_detectable_effect(
    dataset_size: int,
    alpha: float = 0.05,
    beta: float = 0.2,
    k_a: int = 1,
    k_b: int = 1,
    x_a: Optional[list | np.ndarray] = None,
    x_b: Optional[list | np.ndarray] = None,
    var_a: Optional[float] = None,
    var_b: Optional[float] = None,
    omega: Optional[float] = None,
) -> float:
    """
    Find the Minimum Detectable Effect (MDE) for a fixed dataset size.
    with a Type I error rate (alpha) and a Type II error rate (beta).

    See Section 5. of https://arxiv.org/pdf/2411.00640 for more specific details.

    This function can estimate the dataset size in two setups:

    1. From data: you can provide `x_a` and `x_b` to estimate the dataset size required to find the provided MDE.
                  Both `x_a` and `x_b` must be of shape (N,), containing previous evaluation data of models A and B.

    2. From fixed parameters: you can fix `var_a`, `var_b` and `omega` at some sensible level. For instance,
                              `var_a=0`, `var_b=0`, and `omega=1/9` as shown in https://arxiv.org/pdf/2411.00640.

    Therefore, do not provide `x_a` and `x_b` when providing `var_a`, `var_b`, and `omega` (and viceversa).

    Args:
        dataset_size (int): dataset size to find the minimum detectable effect.
        alpha (float): type I error rate. Default to 0.05.
        beta (float): type II error rate. Default to 0.2.
        k_a (int): number of answers from model A that will be sampled in a paired analysis. Default to 1.
        k_b (int): number of answers from model B that will be sampled in a paired analysis. Default to 1.
        x_a (Optional[list | np.ndarray]): previous evaluation data of model A.
        x_b (Optional[list | np.ndarray]): previous evaluation data of model B.
        var_a (Optional[float]): variance of model A scores.
        var_b (Optional[float]): variance of model B scores.
        omega (Optional[float]): defined as `var_a` - `var_b` - 2Cov(`x_a`, `x_b`)

    Returns:
        float: MDE for your dataset size.
    """

    from_data = [x_a, x_b]
    from_params = [var_a, var_b, omega]

    data_provided = all(x is not None for x in from_data)
    params_provided = all(x is not None for x in from_params)

    assert (data_provided and not params_provided) or (
        params_provided and not data_provided
    ), "You must provide either (x_a, x_b) or (var_a, var_b, omega), not both or any."

    if x_a is not None and x_b is not None:
        var_a = np.var(x_a)
        var_b = np.var(x_b)
        cov_ab = np.cov(x_a, x_b)[0, 1]
        omega = var_a + var_b - 2 * cov_ab

    z_alpha = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)

    return np.sqrt(
        (((z_alpha + z_beta) ** 2) * (omega + (var_a / k_a) + (var_b / k_b)))
        / dataset_size
    )


def make_probability_table(
    baseline_acc: float = 0.5,  # accuracy of the baseline model (Model 1/A)
    delta_acc: float = 0.1,  # accuracy difference between Model 1 and Model 2, \theta_B - baseline_acc
    agreement_rate: (
        float | None
    ) = 0.8,  # agreement rate between the two models (Model 1 and Model 2)
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
    if agreement_rate is not None:
        p11 = (agreement_rate + 2 * baseline_acc + delta_acc - 1) / 2
        p10 = baseline_acc - p11
        p01 = baseline_acc + delta_acc - p11
        p00 = agreement_rate - p11
        probs = np.array([p00, p01, p10, p11])
    else:
        acc1 = baseline_acc
        acc2 = baseline_acc + delta_acc
        if acc1 + acc2 <= 1.0:
            # Case A: no overlapping “both correct” mass possible; set p11 = 0
            p11 = 0.0
            p10 = acc1  # Pr(model1 correct, model2 wrong)
            p01 = acc2  # Pr(model1 wrong, model2 correct)
            p00 = 1.0 - acc1 - acc2
        else:
            # Case B: some overlap is forced; set p11 = acc1 + acc2 - 1
            p11 = acc1 + acc2 - 1.0
            p10 = acc1 - p11  # = acc1 - (acc1 + acc2 - 1) = 1 - acc2
            p01 = acc2 - p11  # = acc2 - (acc1 + acc2 - 1) = 1 - acc1
            p00 = 0.0

        probs = np.array([p00, p01, p10, p11], dtype=float)

    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError(
            "The provided parameters do not yield a valid probability table."
        )

    table = probs.reshape(2, 2)
    # some more sanity checks
    assert np.isclose(table[1, :].sum(), baseline_acc), "Error!"
    assert np.isclose(table[:, 1].sum(), baseline_acc + delta_acc), "Error!"
    if agreement_rate is not None:
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
        PowerOutput
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
