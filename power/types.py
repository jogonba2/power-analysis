from dataclasses import dataclass

import numpy as np


@dataclass
class PowerOutput:
    power: float
    mean_eff: float
    type_m: float
    type_s: float


@dataclass
class DGPParameters:
    true_prob_table: np.ndarray
    dataset_size: int

    @property
    def success_probs(self) -> np.ndarray:
        success_model_1 = self.true_prob_table[:, 1].sum()
        success_model_2 = self.true_prob_table[1, :].sum()
        return np.array([success_model_1, success_model_2])


@dataclass
class SimulatedDataset:
    """
    Simulated dataset from a data generating process (DGP).
    Attributes:
        data (np.ndarray): The simulated data.
            Typically this is an array contianing number of correct responses for model 1 and model 2,
            or a 2x2 contingency table (confusion matrix) as follows:
            [
                [num_both_incorrect, num_only_2_correct],
                [num_only_1_correct, num_both_correct]
            ]
        dataset_size (int | np.ndarray): The size of the dataset.
            For a contingency table -- this should be a single integer
            For number of successes  -- this should be an array of two integers, corresponding to number of evals for each model.
    """

    data: np.ndarray
    dataset_size: int | np.ndarray


@dataclass
class StatsTestParameters:
    """
    Base class for parameters for the test statistic function.
    """

    simulated_dataset: SimulatedDataset
    exact: bool = False


@dataclass
class StatsTestOutput:
    p_value: float
    effect: float
