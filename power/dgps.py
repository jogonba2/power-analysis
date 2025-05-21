import catalogue
from numpy.random import Generator

from .types import DGPParameters, SimulatedSample

dgps = catalogue.create("power", "dgps")


@dgps.register("dgp::contingency_table")
def contingency_table(
    dgp_args: DGPParameters, rng: Generator
) -> SimulatedSample:
    data = rng.multinomial(
        dgp_args.dataset_size, dgp_args.true_prob_table.flatten()
    ).reshape(2, 2)
    return SimulatedSample(data=data)
