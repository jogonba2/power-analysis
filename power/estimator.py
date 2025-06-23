from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from power.types import PowerOutput, StatsTestParameters
from sklearn.base import BaseEstimator

class PowerEstimator(BaseEstimator):
    def __init__(self, n_iterations:int=1000, random_state:int=42, n_jobs:int=1):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fitted = False
        self.dgp = None
        self.stat_test = None
        self.effect = None
        self.alpha = 0.05 # should this go here or in fit?

    def fit(self, X, y=None):
        """
        Fit the estimator with the provided dataset.
        """
        self._compute_power(X)
        self.fitted = True
        return self

    def _compute_power(self, X):
        tasks = (
            delayed(PowerEstimator._compute_power_of_single_experiment)(
                true_prob_table=sample["prob_table"],
                dgp=self.dgp,
                test=self.stat_test,
                effect=self.effect,
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
                total=len(X)
                desc=f"TODO add desc"
            )
        )

        print(powers)

    
    @staticmethod
    def _compute_power_for_one_experiment(
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


if __name__ == "__main__":
    # Example usage
    estimator = PowerEstimator(n_iterations=1000, random_state=42, n_jobs=4)
    # Assuming X is a list of dictionaries with keys "prob_table" and "size"
    X = [
        {"prob_table": np.array([[0.8, 0.1], [0.1, 0.8]]), "size": 100},
        {"prob_table": np.array([[0.7, 0.2], [0.2, 0.7]]), "size": 200},
    ]
    estimator.fit(X)
    print("Power estimation completed.")