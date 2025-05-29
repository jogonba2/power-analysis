from functools import partial
from typing import Optional

import numpy as np
from scipy import stats

from .compute_power import compute_power
from .dgps import dgps
from .effects import effects
from .stats_tests import stats_tests
from .types import DGPParameters, PowerOutput


def classification_report(
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
    x_a: Optional[list | np.ndarray] = None,
    x_b: Optional[list | np.ndarray] = None,
    k_a: Optional[int] = None,
    k_b: Optional[int] = None,
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
                  Both `x_a` and `x_b` must be of shape (N,), where each value is a sample of the accuracy
                  with models A and B respectively. For instance, you can generate `x_a` and `x_b` through bootstrapping.

    2. From fixed parameters: you can fix `var_a`, `var_b` and `omega` at some sensible level. For instance,
                              `var_a=0`, `var_b=0`, and `omega=1/9` as shown in https://arxiv.org/pdf/2411.00640.
                              Besides, you must provide `k_a` and `k_b`, i.e., the number of accuracy samples.

    Therefore, do not provide `x_a` and `x_b` when providing `var_a`, `var_b`, `omega`, `k_a`, and `k_b` (and viceversa).

    Args:
        mde (float): minimum detectable effect you want to detect.
        alpha (float): type I error rate. Default to 0.05.
        beta (float): type II error rate. Default to 0.2.
        x_a (Optional[list | np.ndarray]): to estimate variances and omega from data.
        x_b (Optional[list | np.ndarray]): to estimate variances and omega from data.
        k_a (Optional[int]): number of time samples of model A.
        k_b (Optional[int]): number of time samples of model B.
        var_a (Optional[float]): variance of model A accuracy.
        var_b (Optional[float]): variance of model B accuracy.
        omega (Optional[float]): defined as `var_a` - `var_b` - 2Cov(`x_a`, `x_b`)

    Returns:
        int: dataset size required to detect a MDE.
    """

    from_data = [x_a, x_b]
    from_params = [var_a, var_b, omega, k_a, k_b]

    data_provided = all(x is not None for x in from_data)
    params_provided = all(x is not None for x in from_params)

    assert (data_provided and not params_provided) or (
        params_provided and not data_provided
    ), "You must provide either (x_a, x_b) or (var_a, var_b, omega, k_a, k_b), not both or any."

    if x_a is not None and x_b is not None:
        var_a = np.var(x_a)
        var_b = np.var(x_b)
        cov_ab = np.cov(x_a, x_b)[0, 1]
        omega = var_a + var_b - 2 * cov_ab
        k_a = len(x_a)
        k_b = len(x_b)

    z_alpha = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)

    num = ((z_alpha + z_beta) ** 2) * (omega + (var_a / k_a) + (var_b / k_b))

    return num // (mde**2)


def find_minimum_detectable_effect(
    dataset_size: int,
    alpha: float = 0.05,
    beta: float = 0.2,
    x_a: Optional[list | np.ndarray] = None,
    x_b: Optional[list | np.ndarray] = None,
    k_a: Optional[int] = None,
    k_b: Optional[int] = None,
    var_a: Optional[float] = None,
    var_b: Optional[float] = None,
    omega: Optional[float] = None,
) -> float:
    """
    Find the Minimum Detectable Effect (MDE) for a fixed dataset size.
    with a Type I error rate (alpha) and a Type II error rate (beta).

    See Section 5. of https://arxiv.org/pdf/2411.00640 for more specific details.

    This function can estimate the dataset size in two setups:

    1. From data: you can provide `x_a` and `x_b` to estimate the MDE given a dataset size.
                  Both `x_a` and `x_b` must be of shape (N,), where each value is a sample of the accuracy
                  with models A and B respectively. For instance, you can generate `x_a` and `x_b` through bootstrapping.

    2. From fixed parameters: you can fix `var_a`, `var_b` and `omega` at some sensible level. For instance,
                              `var_a=0`, `var_b=0`, and `omega=1/9` as shown in https://arxiv.org/pdf/2411.00640.
                              Besides, you must provide `k_a` and `k_b`, i.e., the number of accuracy samples.

    Therefore, do not provide `x_a` and `x_b` when providing `var_a`, `var_b`, `omega`, `k_a`, and `k_b` (and viceversa).

    Args:
        dataset_size (int): dataset size to find the minimum detectable effect.
        alpha (float): type I error rate. Default to 0.05.
        beta (float): type II error rate. Default to 0.2.
        x_a (Optional[list | np.ndarray]): to estimate variances and omega from data.
        x_b (Optional[list | np.ndarray]): to estimate variances and omega from data.
        k_a (Optional[int]): number of time samples of model A.
        k_b (Optional[int]): number of time samples of model B.
        var_a (Optional[float]): variance of model A accuracy.
        var_b (Optional[float]): variance of model B accuracy.
        omega (Optional[float]): defined as `var_a` - `var_b` - 2Cov(`x_a`, `x_b`)

    Returns:
        float: MDE for your dataset size.
    """

    from_data = [x_a, x_b]
    from_params = [var_a, var_b, omega, k_a, k_b]

    data_provided = all(x is not None for x in from_data)
    params_provided = all(x is not None for x in from_params)

    assert (data_provided and not params_provided) or (
        params_provided and not data_provided
    ), "You must provide either (x_a, x_b) or (var_a, var_b, omega, k_a, k_b), not both or any."

    if x_a is not None and x_b is not None:
        var_a = np.var(x_a)
        var_b = np.var(x_b)
        cov_ab = np.cov(x_a, x_b)[0, 1]
        omega = var_a + var_b - 2 * cov_ab
        k_a = len(x_a)
        k_b = len(x_b)

    z_alpha = stats.norm.ppf(alpha / 2)
    z_beta = stats.norm.ppf(beta)

    return np.sqrt(
        (((z_alpha + z_beta) ** 2) * (omega + (var_a / k_a) + (var_b / k_b)))
        / dataset_size
    )
