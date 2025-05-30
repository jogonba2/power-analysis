import catalogue
import numpy as np
from .types import DGPParameters, SimulatedDataset

effects = catalogue.create("power", "effects")


@effects.register("effect::cohens_g")
def cohens_g(
    dgp_params: DGPParameters | None = None,
    sample: SimulatedDataset | None = None,
) -> float:
    """

    Args:
        true_prob_table (np.ndarray | None): (0, 0) is both incorrect, (1, 1) is both correct,
                                             (0, 1) is model A correct and model B incorrect.
        sample (np.ndarray | None): ...
        dataset_size (int): ...

    Returns:
        float: ...
    """
    assert (dgp_params is None) != (
        sample is None
    ), "Exactly one of 'true_prob_table' or 'sample' must be provided, but not both or neither."
    if dgp_params is not None:
        return (
            dgp_params.true_prob_table[0, 1] - dgp_params.true_prob_table[1, 0]
        )
    else:
        assert sample is not None
        return (sample.data[0, 1] - sample.data[1, 0]) / sample.dataset_size


@effects.register("effect::risk_difference")
def risk_difference(
    dgp_params: DGPParameters | None = None,
    sample: SimulatedDataset | None = None,
) -> float:
    """
    Effect size for an *unpaired* z-test.
    TODO: can we use this https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_effectsize.html ?
    Args:
        true_prob_table (np.ndarray | None): (0, 0) is both incorrect, (1, 1) is both correct,
                                             (0, 1) is model A correct and model B incorrect.
        sample (np.ndarray | None): ...
        dataset_size (int): ...

    Returns:
        float: ...
    """
    assert (dgp_params is None) != (
        sample is None
    ), "Exactly one of 'true_prob_table' or 'sample' must be provided, but not both or neither."

    if dgp_params is not None:
        return dgp_params.success_probs[0] - dgp_params.success_probs[1]
    else:
        assert sample is not None
        assert len(sample.dataset_size) == 2
        return (
            sample.data[0] / sample.dataset_size[0]
            - sample.data[1] / sample.dataset_size[1]
        )
