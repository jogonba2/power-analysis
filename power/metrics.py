from functools import partial

import numpy as np

from .compute_power import compute_power
from .dgps import dgps
from .effects import effects
from .stats_tests import stats_tests
from .types import DGPParameters


def classification_report(
    preds_a: list | np.ndarray,
    preds_b: list | np.ndarray,
    references: list | np.ndarray,
    dgp_fn: str = "contingency_table",
    statistical_test_fn: str = "mcnemar",
    effect_fn: str = "cohens_g",
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
):
    # Create arrays for predictions and references
    if isinstance(preds_a, list):
        preds_a = np.array(preds_a)
    if isinstance(preds_b, list):
        preds_b = np.array(preds_b)
    if isinstance(references, list):
        references = np.array(references)

    # Create the true probability table
    dataset_size = len(references)
    true_prob_table = np.array(
        [
            [
                ((preds_a != references) & (preds_b != references)).mean(),
                ((preds_a == references) & (preds_b != references)).mean(),
            ],
            [
                ((preds_a != references) & (preds_b == references)).mean(),
                ((preds_a == references) & (preds_b == references)).mean(),
            ],
        ]
    )

    # Instantiate the DGP
    dgp_args = DGPParameters(
        true_prob_table=true_prob_table, dataset_size=dataset_size
    )

    data_generating_fn = partial(dgps.get(f"dgp::{dgp_fn}"), dgp_args=dgp_args)

    # Instantiate statistical test and effect estimation
    effect_fn = effects.get(f"effect::{effect_fn}")

    statistical_test_fn = partial(
        stats_tests.get(f"stats_test::{statistical_test_fn}"),
        effect_fn=effect_fn,
    )

    # Prepare the function to compute the *true* effect size for benchmarking
    true_effect_fn = partial(
        effect_fn,
        true_prob_table=true_prob_table,
        sample=None,
        dataset_size=dataset_size,
    )

    # Compute the power
    return compute_power(
        data_generating_fn,
        statistical_test_fn,
        true_effect_fn,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
    )
