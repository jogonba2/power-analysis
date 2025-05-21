import catalogue
import numpy as np

effects = catalogue.create("power", "effects")


@effects.register("effect::cohens_g")
def cohens_g(sample: np.ndarray, dataset_size: int, true_effect: bool):
    """
    TODO: discuss about the name of `sample` and if there are cases we want the true_effect
    to be different from the estimated effect.
    """
    if true_effect:
        return sample[0, 1] - sample[1, 0]
    return (sample[0, 1] - sample[1, 0]) / dataset_size
