from functools import partial
from operator import eq

import datasets as ds
import pandas as pd
import plotnine as pn
from joblib import Parallel, delayed

from power.utils import estimate_power_from_accuracy

DATASETS = {
    "IFEval": 541,
    "BBH": 6511,
    "MATH Lvl 5": 2648,
    "MMLU-PRO": 9497,
    "GPQA": 1252,
}

SIZES = ["small", "medium", "big"]

def compute_row(row):
    return estimate_power_from_accuracy(
        baseline_acc=row.baseline_acc,
        delta_acc=row.delta_acc,
        agreement=0.99,
        dataset_size=int(row.dataset_size),
        alpha=0.05,
        iterations=1000,
    ).__dict__



def load_data():

    dataset = ds.load_dataset("open-llm-leaderboard/contents", split="train")

    dataset = dataset.filter(bool, input_columns="Available on the hub")

    dataset = dataset.filter(partial(eq, "bfloat16"), input_columns="Precision")

    dataset = dataset.filter(
        partial(eq, "Original"), input_columns="Weight type"
    )

    dataset = dataset.map(
        lambda x: {
            "size_cat": SIZES[(x["#Params (B)"] > 1) + (x["#Params (B)"] > 10)]
        }
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

    # 3. compute median‐deviations and pick the closest
    df["median"] = df.groupby(["size_cat", "dataset"])["score"].transform(
        "median"
    )
    df["abs_dev"] = (df["score"] - df["median"]).abs()

    # 4. for each size+dataset, pick the row with minimal abs_dev
    idx = df.groupby(["size_cat", "dataset"])["abs_dev"].idxmin()
    df.loc[idx, ["size_cat", "dataset", "eval_name", "score"]].sort_values(
        ["size_cat", "dataset"]
    )

    # median_performers

    # 3. attach the median for each (size_cat, dataset)
    df["median"] = df.groupby(["size_cat", "dataset"])["score"].transform(
        "median"
    )

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
            pn.aes(x="dataset", y="dev", colour="power_bin", size="abs_dev"),
        )
        # median reference
        + pn.geom_hline(yintercept=0, linetype="dashed")
        # every model as a point
        + pn.geom_jitter(width=0.25, height=0, alpha=0.75)
        # one panel per size bucket
        + pn.facet_wrap("~size_cat", scales="free_y")
        # nicer scales
        + pn.scale_colour_manual(
            name="Worst-case power",
            values={"< 50 %": "#d73027", "50–80 %": "#fee08b", "≥ 80 %": "#1a9850"},
        )
        + pn.scale_size_continuous(name="Δ (%)", range=[1, 4])
        # labels & theme
        + pn.labs(
            title="All models, deviations from size-bucket medians",
            subtitle="Colour = worst-case power, size = abs deviation (percentage points)",
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
        delayed(compute_row)(row) for _, row in df.iterrows()
    )

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

