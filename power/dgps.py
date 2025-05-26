import catalogue
from numpy.random import Generator

from .types import DGPParameters, SimulatedSample

dgps = catalogue.create("power", "dgps")


@dgps.register("dgp::contingency_table")
def contingency_table(
    dgp_args: DGPParameters, rng: Generator
) -> SimulatedSample:
    data = rng.multinomial(
        n=dgp_args.dataset_size,
        pvals=dgp_args.true_prob_table.flatten(),
        size=1,
    ).reshape(2, 2)
    return SimulatedSample(data=data)
