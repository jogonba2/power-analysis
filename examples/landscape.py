from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Import the power analysis utility functions from the package
from power.compute_power import compute_power
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters, PowerOutput


def prob_table_with_agreement(
    baseline_acc: float, delta_acc: float, agreement_rate: float
) -> np.ndarray:
    """
    Given a baseline accuracy, delta accuracy, and agreement_rate,
    returns the 2x2 probability table as:
         [[p_both_incorrect, p_only_2_correct],
          [p_only_1_correct, p_both_correct]]
    This follows the formulas from your original implementation.
    """
    acc1 = baseline_acc
    # this should be acc2??
    acc2 = acc1 + delta_acc
    disagreement_rate = 1.0 - agreement_rate
    if delta_acc > 0:
        p_only_1 = (disagreement_rate - delta_acc) / 2.0
        p_only_2 = (disagreement_rate - delta_acc) / 2.0 + delta_acc
    else:
        p_only_1 = (disagreement_rate + delta_acc) / 2.0 - delta_acc
        p_only_2 = (disagreement_rate + delta_acc) / 2.0
    p_both_correct = acc1 - p_only_1
    p_both_incorrect = 1.0 - p_both_correct - p_only_1 - p_only_2
    return np.array(
        [[p_both_incorrect, p_only_2], [p_only_1, p_both_correct]],
        dtype=np.float32,
    )


def compute_power_of_single_experiment(
    true_prob_table: np.ndarray,
    test: str,
    effect: str,
    dataset_size: int,
    iterations: int,
    alpha: float,
    seed: int,
):
    # Package the true probability table and dataset size into DGPParameters
    dgp_args = DGPParameters(
        true_prob_table=true_prob_table, dataset_size=dataset_size
    )

    try:
        # Selects a specific DGP variant: a contingency table generator
        data_generating_fn = partial(
            dgps.get("dgp::contingency_table"), dgp_args=dgp_args
        )
    except ValueError:
        return PowerOutput(np.nan, np.nan, np.nan, np.nan)

    # Retrieve the function to compute Cohen's g as the effect size
    effect_fn = effects.get(effect)

    # Create a function for the McNemar test
    statistical_test_fn = partial(stats_tests.get(test), effect_fn=effect_fn)

    # Prepare the function to compute the *true* effect size for benchmarking
    true_effect_fn = partial(
        effect_fn,
        true_prob_table=true_prob_table,
        sample=None,
        dataset_size=dataset_size,
    )

    # Compute statistical power and related metrics using simulation
    power = compute_power(
        data_generating_fn,
        statistical_test_fn,
        true_effect_fn,
        iterations,
        alpha,
        seed,
    )

    return power


def sample(
    n: int,
    # what are these??
    baseline_range=(0.5, 0.9),
    delta_range=(0.01, 0.5),
    agreement_range=(0.5, 0.99),
    ds_sizes=(500, 1000, 2000),
    seed=123,
):
    # Why the brute force?
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        p00, p01, p10, p11 = rng.dirichlet([1, 1, 1, 1])
        b = p11 + p10
        d = (p11 + p01) - b
        agr = p11 + p00

        # enforce your desired marginal ranges
        if not (baseline_range[0] <= b <= baseline_range[1]):
            continue
        if not (delta_range[0] <= d <= delta_range[1]):
            continue
        if not (agreement_range[0] <= agr <= agreement_range[1]):
            continue

        s = int(rng.choice(ds_sizes))
        samples.append((b, d, s, agr))
    return samples


def make_probability_table(
    baseline_acc: float = 0.5,  # accuracy of the baseline model (Model 1/A)
    delta_acc: float = 0.1,  # accuracy difference between Model 1 and Model 2, \theta_B - baseline_acc
    agreement_rate: float = 0.8,  # agreement rate between the two models (Model 1 and Model 2)
):
    """
    Returns a 2x2 probability table representing the joint distribution of two models predictions.
    The table is structured as follows:
        [
            [p_both_incorrect, p_only_2_correct],
            [p_only_1_correct, p_both_correct]
        ]

    where:
      - the baseline model is model 1, so that accuracy_model1 = table[1, :].sum() = baseline_acc
      - the second model, model 2, has accuracy_model2 = table[:, 1].sum() = baseline_acc + delta_acc
      - the agreement rate is the sum of the probabilities of both models giving the same prediction, i.e.,
            table.diagonal.sum() = agreement_rate
    """
    # p11 is the probability that both models are correct
    p11 = (agreement_rate + 2 * baseline_acc + delta_acc - 1) / 2
    p10 = baseline_acc - p11
    p01 = baseline_acc + delta_acc - p11
    p00 = agreement_rate - p11
    probs = np.array([p00, p01, p10, p11])
    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("Infeasible combination.")
    probs_table = probs.reshape(2, 2)
    assert (
        probs_table[1, :].sum() == baseline_acc
        and probs_table[:, 1].sum() == baseline_acc + delta_acc
        and probs_table.diagonal().sum() == agreement_rate
    ), "Something went wrong with the probabilities...."
    return probs_table


if __name__ == "__main__":
    # parameters
    num_points = 10000
    iterations = 50000
    alpha = 0.05
    seed = 123

    samples = sample(num_points, seed=seed)

    test_effects = [
        ("stats_test::mcnemar", "effect::cohens_g"),
        ("stats_test::unpaired_z", "effect::risk_difference"),
    ]

    dfs = []
    for test, effect in test_effects:

        tasks = (
            delayed(compute_power_of_single_experiment)(
                prob_table_with_agreement(baseline, delta, agreement),
                test,
                effect,
                size,
                iterations,
                alpha,
                seed,
            )
            for idx, (baseline, delta, size, agreement) in enumerate(samples)
        )

        powers = Parallel(n_jobs=-1, backend="multiprocessing")(
            tqdm(tasks, total=num_points)
        )

        df = pd.DataFrame(powers)

        df["test"] = test

        dfs.append(df)

    samples = pd.DataFrame(
        samples, columns=["baseline", "delta", "size", "agreement"]
    )

    df = pd.concat(
        [pd.concat([samples, samples], axis=0), pd.concat(dfs, axis=0)], axis=1
    )

    df = df.pivot(
        index=["baseline", "delta", "size", "agreement"],
        columns="test",
        values=["power", "mean_eff", "type_m", "type_s"],
    )

    df.to_csv("power-pivot.csv")
