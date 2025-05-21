

# Usage example

from functools import partial
import numpy as np

# Import the power analysis utility functions from the package
from power.compute_power import compute_power 
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters


if __name__ == "__main__":
    # Define a 2x2 contingency table representing true conditional probabilities
    true_prob_table = np.array([[0.0, 0.5], [0.3, 0.2]], dtype="float32")
    
    # Set the size of each synthetic dataset
    dataset_size = 1000

    # Package the true probability table and dataset size into DGPParameters
    dgp_args = DGPParameters(
        true_prob_table=true_prob_table, dataset_size=dataset_size
    )

    # Selects a specific DGP variant: a contingency table generator
    data_generating_fn = partial(
        dgps.get("dgp::contingency_table"), dgp_args=dgp_args
    )

    # Retrieve the function to compute Cohen's g as the effect size
    effect_fn = effects.get("effect::cohens_g")

    # Create a function for the McNemar test
    statistical_test_fn = partial(
        stats_tests.get("stats_test::mcnemar"), effect_fn=effect_fn
    )

    # Prepare the function to compute the *true* effect size for benchmarking
    true_effect_fn = partial(
        effect_fn,
        true_prob_table=true_prob_table,
        sample=None,
        dataset_size=dataset_size,
    )

    # Define experiment parameters
    # Number of simulation iterations
    iterations = 11
    # Significance level
    alpha = 0.05
    seed = 13

    # Compute statistical power and related metrics using simulation
    power = compute_power(
        data_generating_fn,
        statistical_test_fn,
        true_effect_fn,
        iterations,
        alpha,
        seed,
    )

    # Verify that the statistical power is 100%
    assert power.power == 1.0

    # Validate the computed mean effect size is as expected (Cohen's g ≈ 0.193)
    np.testing.assert_almost_equal(power.mean_eff, 0.193363, decimal=4)

    # Validate the computed Type M error (magnitude exaggeration)
    np.testing.assert_almost_equal(power.type_m, 0.966818, decimal=4)

    # Validate the computed Type S error (sign errors, here expected to be zero)
    np.testing.assert_almost_equal(power.type_s, 0.0, decimal=0)