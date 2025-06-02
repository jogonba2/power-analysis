from functools import partial
from operator import eq

import datasets as ds
import pandas as pd
import plotnine as pn
from joblib import Parallel, delayed
from tqdm import tqdm

from power.utils import estimate_power_from_accuracy

DATASETS = {
    "BBH": 6511,
    "GPQA": 1252,
    "IFEval": 541,
    "MATH Lvl 5": 2648,
    "MMLU-PRO": 9497,
}

SIZES = ["base", "medium", "large"]

NUM_ITERATIONS_PER_SAMPLE = 1_000

AGREEMENT = 0.8


def compute_row(row):
    return estimate_power_from_accuracy(
        baseline_acc=row.baseline_acc,
        delta_acc=row.delta_acc,
        agreement=None,
        dataset_size=int(row.dataset_size),
        alpha=0.05,
        iterations=NUM_ITERATIONS_PER_SAMPLE,
    ).__dict__


def load_data():

    dataset = ds.load_dataset("open-llm-leaderboard/contents", split="train")

    dataset = dataset.filter(bool, input_columns="Available on the hub")

    dataset = dataset.filter(partial(eq, "bfloat16"), input_columns="Precision")

    dataset = dataset.filter(
        partial(eq, "Original"), input_columns="Weight type"
    )

    dataset = dataset.map(
        lambda params: {"size_cat": SIZES[(params > 1) + (params > 10)]},
        input_columns="#Params (B)",
    )

    df = (
        dataset.to_pandas()
        .reset_index()
        .melt(  # keep original row‐idx
            id_vars=["index", "eval_name", "size_cat"],
            value_vars=DATASETS.keys(),
            var_name="dataset",
            value_name="score",
        )
    )

    df["dataset_size"] = df["dataset"].map(DATASETS)

    df["median"] = df.groupby(["size_cat", "dataset"])["score"].transform(
        "median"
    )
    df["abs_dev"] = (df["score"] - df["median"]).abs()

    df["baseline_acc"] = df["median"] / 100

    df["delta_acc"] = (df["score"] - df["median"]) / 100

    return df.dropna()


def plot_stats(df):
    (
        pn.ggplot(df, pn.aes(x="dataset", y="score"))
        + pn.geom_boxplot(alpha=0.7, outlier_size=1)
        + pn.facet_wrap("~size_cat", scales="free_y", nrow=1)
        + pn.labs(
            title="Distribution of Model Scores by Dataset and Size Category",
            x="Dataset",
            y="Score",
            fill="Size Category",
        )
        + pn.theme(
            axis_text_x=pn.element_text(rotation=45, hjust=1),
            legend_position="bottom",
        )
    ).save("examples/post-hoc-stats.png")


def plot_power(df):
    p = (
        pn.ggplot(
            df,
            pn.aes(x="dataset", y="dev", colour="power_bin"),
        )
        # median reference
        + pn.geom_hline(yintercept=0, linetype="dashed")
        # every model as a point
        + pn.geom_jitter(width=0.25, height=0.01, alpha=0.75)
        # one panel per size bucket
        + pn.facet_wrap("~size_cat", scales="free_y")
        # nicer scales
        + pn.scale_colour_manual(
            name="power",
            values={
                "< 50 %": "#d73027",
                "50–80 %": "#fee08b",
                "≥ 80 %": "#1a9850",
            },
        )
        # labels & theme
        + pn.labs(
            title="Power Analysis of Open LLM Benchmark",
            x="",
            y="Δ",
        )
        + pn.theme(
            figure_size=(14, 5),
            axis_text_x=pn.element_text(rotation=45, ha="right"),
            legend_position="right",
            legend_title=pn.element_text(size=9),
            legend_text=pn.element_text(size=8),
        )
    )

    p.save("examples/post-hoc-analysis.png")


if __name__ == "__main__":
    df = load_data()

    plot_stats(df)

    # df = openllm.groupby(["size_cat", "dataset"]).sample(n=100, random_state=42)

    # Parallel loop over DataFrame rows
    results = Parallel(n_jobs=-1)(
        delayed(compute_row)(row)
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Estimating power"
        )
    )
    tqdm(df.iterrows(), total=len(df), desc="Estimating power")

    # # concat results
    df = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)

    df["dev"] = df["score"] - df["median"]  # Δ score (pp)

    df["power_bin"] = pd.cut(
        df["power"],
        bins=[0, 0.5, 0.8, 1.0],
        labels=["< 50 %", "50–80 %", "≥ 80 %"],
    )

    df = df.dropna()

    plot_power(df)
