import catalogue
import numpy as np

effects = catalogue.create("power", "effects")


@effects.register("effect::cohens_g")
def cohens_g(
    true_prob_table: np.ndarray | None,
    sample: np.ndarray | None,
    dataset_size: int,
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
    assert (true_prob_table is None) != (
        sample is None
    ), "Exactly one of 'true_prob_table' or 'sample' must be provided, but not both or neither."
    if true_prob_table is not None:
        return true_prob_table[0, 1] - true_prob_table[1, 0]
    return (sample[0, 1] - sample[1, 0]) / dataset_size


@effects.register("effect::risk_difference")
def risk_difference(
    true_prob_table: np.ndarray | None,
    sample: np.ndarray | None,
    dataset_size: int,
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
    assert (true_prob_table is None) != (
        sample is None
    ), "Exactly one of 'true_prob_table' or 'sample' must be provided, but not both or neither."
    data = true_prob_table if true_prob_table is not None else sample
    number_of_successes_model_1 = data[:, 1].sum()
    number_of_successes_model_2 = data[1:, :].sum()
    diff = number_of_successes_model_1 - number_of_successes_model_2
    return diff / dataset_size
