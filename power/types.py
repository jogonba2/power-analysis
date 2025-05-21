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


@dataclass
class SimulatedSample:
    data: np.ndarray


@dataclass
class StatsTestParameters:
    """
    Base class for parameters for the test statistic function.
    """

    simulated_sample: SimulatedSample
    exact: bool

    @property
    def dataset_size(self):
        return np.sum(self.simulated_sample.data)


@dataclass
class StatsTestOutput:
    p_value: float
    effect: float
