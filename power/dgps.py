import catalogue
import numpy as np
from numpy.random import Generator

from .types import DGPParameters, SimulatedDataset

dgps = catalogue.create("power", "dgps")


@dgps.register("dgp::contingency_table")
def contingency_table(
    dgp_params: DGPParameters, rng: Generator
) -> SimulatedDataset:
    data = rng.multinomial(
        n=dgp_params.dataset_size,
        pvals=dgp_params.true_prob_table.flatten(),
        size=1,
    ).reshape(2, 2)
    return SimulatedDataset(data=data, dataset_size=dgp_params.dataset_size)


@dgps.register("dgp::successes_and_failures")
def successes_and_failures(
    dgp_params: DGPParameters, rng: Generator
) -> SimulatedDataset:
    nobs_per_class = np.array([dgp_params.dataset_size] * 2)
    data = rng.binomial(
        n=nobs_per_class,
        p=dgp_params.success_probs,
    )
    return SimulatedDataset(data=data, dataset_size=nobs_per_class)
