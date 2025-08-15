import numpy as np
import pandas as pd
import plotnine as pn

SOURCE = "tests/simulations.csv.gz"


def debug_plot(df):
    df = df[
        df.test.isin(
            [
                "stats_test::unpaired_z",
                "stats_test::mcnemar",
                "stats_test::paired_z",
                "stats_test::paired_t",
            ]
        )
    ]
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
        + pn.facet_wrap("size", ncol=3)
        + pn.scale_x_continuous(limits=(0, 0.2), breaks=[0.1, 0.2])
        + pn.labs(x="Δ accuracy", y="Estimated power", color="N")
        + pn.theme(
            legend_position="bottom",
            legend_text=pn.element_text(size=12),
            legend_title=pn.element_text(size=13),
            legend_box_margin=5,
            legend_background=pn.element_rect(color="black", size=1, alpha=1),
            panel_border=pn.element_rect(color="black", fill=None, size=1.2),
            axis_text=pn.element_text(size=11, weight="bold"),
            axis_title=pn.element_text(size=14),
            panel_background=pn.element_rect(fill="white"),
            panel_grid_major=pn.element_line(color="#F0F0F0"),
            axis_ticks_length=6,
            axis_ticks_major=pn.element_line(size=2),
        )
        + pn.guides(color=pn.guide_legend(reverse=True))
    ).save("img/test-comparison.png", width=12, height=8, units="in", dpi=300)


df = pd.read_csv(SOURCE, compression="gzip", index_col=0)

df["agreement"] = pd.cut(df.agreement, bins=2, labels=["low", "high"])

debug_plot(df)

df = df[["power", "size", "agreement", "delta", "baseline", "test"]]

df = df.dropna()

mcnmear = df[df.test == "stats_test::mcnemar"]

(
    pn.ggplot(mcnmear, pn.aes(x="delta", y="power", color="factor(size)"))
    + pn.stat_summary(fun_y=np.mean, geom="line", size=1, alpha=0.9)
    + pn.geom_hline(yintercept=0.8, linetype="dashed", color="grey")
    + pn.facet_wrap("~ agreement")
    + pn.scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.50, 0.75, 1.0])
    + pn.labs(
        x="Δ accuracy", y="Estimated power", title="Power vs Δ", color="N"
    )
    + pn.scale_color_cmap_d(
        cmap_name="tab10",
        name="N",
    )
    + pn.guides(color=pn.guide_legend(reverse=True))
).save("img/fig3-cardetal-mcnmear.png", width=5, height=5, units="in", dpi=300)

(
    pn.ggplot(mcnmear, pn.aes(x="delta", y="power", color="factor(size)"))
    + pn.stat_summary(fun_y=np.mean, geom="line", size=1.5, alpha=0.99)
    + pn.geom_hline(yintercept=0.8, linetype="dashed", color="grey")
    + pn.scale_y_continuous(
        limits=(0, 1), breaks=[0, 0.20, 0.40, 0.60, 0.80, 1.0]
    )
    + pn.scale_x_continuous(limits=(0.04, 0.20))
    + pn.labs(x="Δ accuracy", y="Estimated power", color="N")
    + pn.scale_color_cmap_d(
        cmap_name="Set1",
        drop=True,
        name="N",
    )
    + pn.guides(color=pn.guide_legend(reverse=False))
    + pn.theme(
        legend_position="bottom",
        legend_text=pn.element_text(size=12),
        legend_title=pn.element_text(size=13),
        legend_box_margin=5,
        legend_background=pn.element_rect(color="black", size=1, alpha=1),
        panel_border=pn.element_rect(color="black", fill=None, size=1.2),
        axis_text=pn.element_text(size=11, weight="bold"),
        axis_title=pn.element_text(size=14),
        panel_background=pn.element_rect(fill="white"),
        panel_grid_major=pn.element_line(color="#F0F0F0"),
        axis_ticks_length=6,
        axis_ticks_major=pn.element_line(size=2),
    )
).save(
    "img/mcnemar-power-vs-delta-without-agreement.png",
    width=6,
    height=6,
    units="in",
    dpi=300,
)


# test vs test
(
    pn.ggplot(df, pn.aes(x="delta", y="power", color="test"))
    + pn.stat_summary(fun_y=np.mean, geom="line", size=1, alpha=0.9)
    + pn.geom_hline(yintercept=0.8, linetype="dashed", color="grey")
    + pn.scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.50, 0.75, 1.0])
    + pn.labs(
        x="Δ accuracy", y="Estimated power", title="Power vs Δ", color="test"
    )
).save("img/compare-all-tests.png")
