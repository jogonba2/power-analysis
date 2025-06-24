from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from tqdm import tqdm
from tqdm.auto import tqdm

# Import the power analysis utility functions from the package
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters, PowerOutput, StatsTestParameters

from .dgps import dgps
from .effects import effects
from .stats_tests import stats_tests


class PowerEstimator(BaseEstimator):
    def __init__(
        self,
        dgp: str,
        stat_test: str,
        effect: str,
        n_iterations: int = 1000,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        self.dgp_name = dgp
        self.stat_test_name = stat_test
        self.effect_name = effect
        self.dgp = dgps.get(dgp)
        self.stat_test = stats_tests.get(stat_test)
        self.effect = effects.get(effect)
        self.alpha = 0.05  # should this go here or in fit?
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit the estimator with the provided dataset.
        """

        tasks = (
            delayed(PowerEstimator._compute_power_of_single_experiment)(
                true_prob_table=sample["prob_table"],
                dgp=self.dgp,
                stat_test=self.stat_test,
                effect_fn=self.effect,
                dataset_size=sample["size"],
                iterations=self.n_iterations,
                alpha=self.alpha,
                seed=self.random_state + i,
            )
            for i, sample in enumerate(X)
        )

        powers = Parallel(n_jobs=-1, backend="multiprocessing")(
            tqdm(
                tasks,
                total=len(X),
                desc=f"Estimating power using {self.dgp_name} DGP, {self.stat_test_name} test, and {self.effect_name} effect",
            )
        )
        self.landscape_ = pd.concat(
            [pd.DataFrame(X), pd.DataFrame(powers)], axis=1
        ).dropna()

        if self.landscape_.empty:
            raise ValueError(
                "No valid power estimates were computed. Check your input data."
            )

        self.fitted = True

        return self

    @staticmethod
    def _compute_power_of_single_experiment(
        true_prob_table: np.ndarray,
        dgp: callable,
        stat_test: callable,
        effect_fn: callable,
        dataset_size: int,
        iterations: int,
        alpha: float,
        seed: int,
    ):
        # Package the true probability table and dataset size into DGPParameters
        dgp_params = DGPParameters(
            true_prob_table=true_prob_table, dataset_size=dataset_size
        )

        try:
            data_generating_fn = partial(dgp, dgp_params=dgp_params)
        except ValueError:
            return PowerOutput(np.nan, np.nan, np.nan, np.nan)

        # Create a function for the McNemar test
        statistical_test_fn = partial(stat_test, effect_fn=effect_fn)

        # Prepare the function to compute the *true* effect size for benchmarking
        true_effect_fn = partial(effect_fn, dgp_params=dgp_params)

        # Compute statistical power and related metrics using simulation
        power = PowerEstimator._compute_power(
            data_generating_fn,
            statistical_test_fn,
            true_effect_fn,
            iterations,
            alpha,
            seed,
        )

        return power

    @staticmethod
    def _compute_power(
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
                [
                    abs(effect) / abs(true_effect)
                    for effect in significant_effects
                ]
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
