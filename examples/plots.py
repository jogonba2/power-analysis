import numpy as np
import pandas as pd
import plotnine as pn

df = pd.read_csv("tests/results.csv.gz", compression="gzip", index_col=0)

df["power"] = df.power.apply(eval)

df["power"] = df.power.apply(lambda p: p[0] if not isinstance(p, float) else p)

df["agreement"] = pd.cut(df.agreement, bins=2, labels=["low", "high"])

df = df[["power", "size", "agreement", "delta", "baseline", "test"]]

df = df.dropna()

mcnmear = df[df.test == 'stats_test::mcnemar']

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
).save("fig3-cardetal-mcnmear.png")


unpaired = df[df.test == 'stats_test::unpaired_z']
(
    pn.ggplot(unpaired, pn.aes(x="delta", y="power", color="factor(size)"))
    + pn.stat_summary(fun_y=np.mean, geom="line", size=1, alpha=0.9)
    + pn.stat_summary(fun_y=np.mean, geom="point", size=1, alpha=0.5)
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
).save("fig3-cardetal-unpaired.png")

# test vs test
(
    pn.ggplot(df, pn.aes(x="delta", y="power", color="test"))
    + pn.stat_summary(fun_y=np.mean, geom="line", size=1, alpha=0.9)
    + pn.geom_hline(yintercept=0.8, linetype="dashed", color="grey")
    + pn.scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.50, 0.75, 1.0])
    + pn.labs(
        x="Δ accuracy", y="Estimated power", title="Power vs Δ", color="test"
    )
).save("test-vs-test.png")
