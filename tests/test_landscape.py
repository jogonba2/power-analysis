from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from tqdm.auto import tqdm

# Import the power analysis utility functions from the package
from power.compute_power import compute_power
from power.dgps import dgps
from power.effects import effects
from power.stats_tests import stats_tests
from power.types import DGPParameters, PowerOutput


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


def make_probability_table(
    baseline_acc: float = 0.5,  # accuracy of the baseline model (Model 1/A)
    delta_acc: float = 0.1,  # accuracy difference between Model 1 and Model 2, \theta_B - baseline_acc
    agreement_rate: float = 0.8,  # agreement rate between the two models (Model 1 and Model 2)
) -> np.ndarray:
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
        raise ValueError(
            "The provided parameters do not yield a valid probability table."
        )

    table = probs.reshape(2, 2)
    # some more sanity checks
    assert np.isclose(table[1, :].sum(), baseline_acc), "Error!"
    assert np.isclose(table[:, 1].sum(), baseline_acc + delta_acc), "Error!"
    assert np.isclose(table.diagonal().sum(), agreement_rate), "Error!"

    return table


def test_landscape_code():

    # parameters
    num_simulations_per_sample = 1000
    alpha = 0.05
    seed = 123

    baseline_values = np.linspace(0.5, 0.9, 10)
    delta_values = np.linspace(0.01, 0.3, 50)
    agreement_values = np.linspace(0.0, 0.99, 50)
    dataset_sizes = [50, 100, 500, 1000]
    grid_for_samples = product(
        baseline_values, delta_values, agreement_values, dataset_sizes
    )
    probability_tables = []
    for baseline, delta, agreement, size in grid_for_samples:
        try:  # skip infeasible combinations
            table = make_probability_table(baseline, delta, agreement)
        except ValueError:
            continue
        probability_tables.append(
            {
                "size": size,
                "baseline": baseline,
                "delta": delta,
                "agreement": agreement,
                "prob_table": table,
            }
        )
    print(f"{len(probability_tables)} valid probability tables generated.")
    test_effects = [
        ("stats_test::mcnemar", "effect::cohens_g"),
        ("stats_test::unpaired_z", "effect::risk_difference"),
        ("sanity_check::unpaired_z", None),  # sanity check
    ]

    dfs = []
    for test, effect in test_effects:
        if test == "sanity_check::unpaired_z":
            # sanity check
            powers = [
                NormalIndPower(ddof=1).power(
                    effect_size=proportion_effectsize(
                        prop1=sample["prob_table"][0, 1],
                        prop2=[sample["prob_table"][1, 0]],
                    ),
                    nobs1=sample["size"],
                    alpha=alpha,
                    ratio=1,
                    alternative="two-sided",
                )
                for sample in probability_tables
            ]
            df = pd.DataFrame({"power": powers})
        else:
            tasks = (
                delayed(compute_power_of_single_experiment)(
                    sample["prob_table"],
                    test,
                    effect,
                    sample["size"],
                    num_simulations_per_sample,
                    alpha,
                    seed,
                )
                for sample in probability_tables
            )

            powers = Parallel(n_jobs=-1, backend="multiprocessing")(
                tqdm(
                    tasks,
                    total=len(probability_tables),
                    desc=f"Running {test} with {effect}",
                )
            )

            df = pd.DataFrame(powers)
        df["test"] = test
        dfs.append(df)

    ### tidy up and plot ###
    results = pd.concat(
        [
            pd.concat(
                [
                    pd.DataFrame(probability_tables),
                ]
                * len(test_effects),
                axis=0,
            ),
            pd.concat(dfs, axis=0),
        ],
        axis=1,
    )

    #
    averaged = results.groupby(["delta", "test"], as_index=False)[
        "power"
    ].mean()

    # Plotting the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for test_name, g in averaged.groupby("test"):
        plt.plot(g["delta"], g["power"], marker="o", label=test_name)

    plt.xlabel("Delta (accuracy difference)")
    plt.ylabel("Statistical power")
    plt.title("Power vs Δ (averaged over baseline, agreement, size)")
    plt.legend(title="Test")
    plt.grid(True)
    plt.tight_layout()

    # plot the different dataset sizes using subplots and shared axes
    fig, axes = plt.subplots(
        nrows=1, ncols=4, figsize=(12, 3), sharex=True, sharey=True
    )
    for ax, size in zip(axes, results["size"].unique()):
        subset = results[results["size"] == size]
        # average across the baseline and agreement parameters
        subset = subset.groupby(["delta", "test"], as_index=False).mean()
        for test_name, g in subset.groupby("test"):
            ax.plot(
                g["delta"],
                g["power"],
                # marker="o",
                label=test_name,
                # alpha=0.7,
            )
        ax.set_title(f"Dataset size: {size}")
        ax.grid(True)
    # axes[0].set_ylabel("Estimated power")
    # x axis title for the entire figure

    # ax.legend(title="Test")
    # global labels
    fig.text(0.5, -0.03, "Δ (accuracy difference)", ha="center")  # x-axis
    fig.text(
        -0.01, 0.5, "Estimated power", va="center", rotation="vertical"
    )  # y-axis

    # one legend (take handles/labels from the last axis that was plotted)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Test",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(labels),
    )

    plt.suptitle("Power vs Δ (by dataset size)")
    plt.tight_layout(rect=[0, 0, 1, 1])  # adjust to fit title
    plt.savefig("tests/debug-unpaired-z.png")
