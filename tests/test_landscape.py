from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import plotnine as pn
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
    dgp: str,
    test: str,
    effect: str,
    dataset_size: int,
    iterations: int,
    alpha: float,
    seed: int,
):
    # Package the true probability table and dataset size into DGPParameters
    dgp_params = DGPParameters(
        true_prob_table=true_prob_table, dataset_size=dataset_size
    )

    try:
        data_generating_fn = partial(dgps.get(dgp), dgp_params=dgp_params)
    except ValueError:
        return PowerOutput(np.nan, np.nan, np.nan, np.nan)

    # Retrieve the function to compute Cohen's g as the effect size
    effect_fn = effects.get(effect)

    # Create a function for the McNemar test
    statistical_test_fn = partial(stats_tests.get(test), effect_fn=effect_fn)

    # Prepare the function to compute the *true* effect size for benchmarking
    true_effect_fn = partial(effect_fn, dgp_params=dgp_params)

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
    num_simulations_per_sample = 1_000
    alpha = 0.05
    seed = 20250530

    baseline_values = np.linspace(0.5, 0.9, 20)
    delta_values = np.linspace(0.01, 0.2, 20)
    agreement_values = np.linspace(0.0, 0.99, 20)
    dataset_sizes = [10, 20, 50, 100, 500]
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
    dgp_test_effects = [
        ("dgp::contingency_table", "stats_test::mcnemar", "effect::cohens_g"),
        (
            "dgp::successes_and_failures",
            "stats_test::unpaired_z",
            "effect::risk_difference",
        ),
        (None, "sanity_check::unpaired_z", None),  # sanity check
    ]

    dfs = []
    for dgp, test, effect in dgp_test_effects:
        if test == "sanity_check::unpaired_z":
            # sanity check
            powers = [
                # ddof is 0 by default, let's keep it like this
                NormalIndPower(ddof=0).power(
                    # from docs: "standardized effect size, difference between the two means divided
                    # by the standard deviation. effect size has to be positive"
                    effect_size=proportion_effectsize(
                        prop1=sample["prob_table"][:, 1].sum(),
                        prop2=sample["prob_table"][1:, :].sum(),
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
                    true_prob_table=sample["prob_table"],
                    dgp=dgp,
                    test=test,
                    effect=effect,
                    dataset_size=sample["size"],
                    iterations=num_simulations_per_sample,
                    alpha=alpha,
                    seed=seed + i,
                )
                for i, sample in enumerate(probability_tables)
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
                * len(dgp_test_effects),
                axis=0,
            ),
            pd.concat(dfs, axis=0),
        ],
        axis=1,
    )

    results.to_csv("tests/results.csv.gz", compression="gzip")


def test_plots():
    df = pd.read_csv("tests/results.csv.gz", compression="gzip", index_col=0)
    df["prob_table"] = df.prob_table.apply(
        lambda s: np.fromstring(
            s.replace("\n", " ").replace("[", "").replace("]", ""), sep=" "
        ).reshape(-1, 2)
    )

    df = df[df.delta <= 0.2]

    df = df[["power", "size", "agreement", "delta", "baseline", "test"]]

    df = df.dropna()

    (
        pn.ggplot(df, pn.aes(x="delta", y="power", color="test"))
        + pn.stat_summary(fun_y=np.mean, geom="line", size=1, alpha=0.9)
        + pn.geom_hline(yintercept=0.8, linetype="dashed", color="grey")
        + pn.facet_grid(cols="size")
        + pn.scale_x_continuous(limits=(0, 0.2), breaks=[0.1, 0.2])
        + pn.labs(
            x="Δ accuracy", y="Estimated power", title="Power vs Δ", color="N"
        )
        + pn.theme_seaborn()
        + pn.theme(
            legend_direction="horizontal"
        )  # Makes it a horizontal legend
        + pn.theme(legend_position="top")
        + pn.guides(color=pn.guide_legend(reverse=True))
    ).save("debug-unpaired-z-2.png", width=12, height=8, units="in", dpi=300)
