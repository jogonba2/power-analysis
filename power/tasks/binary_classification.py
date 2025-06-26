from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance

from power.tasks import Task
from power.types import PowerOutput

from ..estimator import PowerEstimator
from ..types import PowerOutput


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


def make_dataset(
    n_instances: int | list,
    baseline_acc: float | list,
    delta_acc: float | list,
    agreement_rate: float | list | None = None,
    seed: int = 42,
) -> list[dict]:

    grid_for_samples = product(
        np.atleast_1d(baseline_acc),
        np.atleast_1d(delta_acc),
        np.atleast_1d(agreement_rate) if agreement_rate is not None else [None],
        np.atleast_1d(n_instances),
    )

    dataset = []

    for baseline, delta, agreement, size in grid_for_samples:
        try:  # skip infeasible combinations
            table = make_probability_table(baseline, delta, agreement)
        except ValueError:
            continue
        dataset.append(
            {
                "size": size,
                "baseline": baseline,
                "delta": delta,
                "agreement": agreement,
                "prob_table": table,
            }
        )

    return dataset


class BinaryClassification(Task):
    def __init__(
        self, n_iterations: int = 1000, n_jobs: int = 1, random_state: int = 42
    ):
        self.power_estimator = PowerEstimator(
            dgp="dgp::contingency_table",
            stat_test="stats_test::mcnemar",
            effect="effect::cohens_g",
            n_iterations=n_iterations,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit the task with the provided dataset.
        """
        self.power_estimator.fit(X)
        self.fitted = True
        self.landscape_: pd.DataFrame = self.power_estimator.landscape_

    def predict_from_score(
        self,
        baseline_acc: float,
        delta_acc: float,
        dataset_size: int,
        agreement: float,
    ) -> PowerOutput:
        if not self.fitted:
            raise ValueError(
                "The model must be fitted before making predictions."
            )
        # given the landscape, find the closest scenario to the one given here
        query = np.array([baseline_acc, delta_acc, agreement, dataset_size])

        # find the closest scenario in the landscape
        distances = self.landscape_[
            ["baseline", "delta", "agreement", "size"]
        ].apply(lambda row: distance.euclidean(row, query), axis=1)

        closest_index = distances.idxmin()

        return self.landscape_.iloc[closest_index]

    def predict_from_predictions() -> PowerOutput: ...

    def predict_mde(
        self,
        n_instances: int,
        baseline: float,
        agreement: float,
        power: float = 0.8,
    ) -> PowerOutput:
        # TODO: standardize use of n_instances and size
        # find mde, given n_instances, baseline_acc and agreement
        if not self.fitted:
            raise ValueError(
                "The model must be fitted before making predictions."
            )

        query = "(size <= @n_instances) & (baseline <= @baseline) & (agreement <= @agreement) & (power >= @power)"

        results = self.landscape_.query(expr=query)

        results = results.sort_values(by="delta", ascending=False)

        if results.empty:
            return None

        top_result = results.iloc[0]

        return PowerOutput(
            mde=top_result["delta"],
            baseline=top_result["baseline"],
            agreement=top_result["agreement"],
            size=top_result["size"],
            power=top_result["power"],
        )

    def predict_n(
        self,
        baseline: float,
        delta: float,
        agreement: float,
        mde: float,
        power: float = 0.8,
    ) -> int:
        # find n, given mde, baseline_acc and agreement
        if not self.fitted:
            raise ValueError(
                "The model must be fitted before making predictions."
            )

        query = "baseline <= @baseline & delta <= @mde & agreement <= @agreement & power >= @power"

        results = self.landscape_.query(expr=query)

        results = results.sort_values(by="size", ascending=False)

        if results.empty:
            return None

        return results.iloc[0]["size"]

    def _find_dataset_size(
        self,
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

        num = ((z_alpha + z_beta) ** 2) * (
            omega + (var_a / k_a) + (var_b / k_b)
        )

        return num // (mde**2)

    def _find_minimum_detectable_effect(
        self,
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
            (
                ((z_alpha + z_beta) ** 2)
                * (omega + (var_a / k_a) + (var_b / k_b))
            )
            / dataset_size
        )
