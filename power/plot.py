import pandas as pd
from plotnine import *

from .types import PowerBounds


def plot_bounds(data: PowerBounds) -> ggplot:
    df = pd.DataFrame(
        {
            "metric": ["power", "mean", "typeM", "typeS"] * 2,
            "group": ["Upper Bound"] * 4 + ["Lower Bound"] * 4,
            "value": [
                data.upper.power,
                data.upper.mean_eff,
                data.upper.type_m,
                data.upper.type_s,
                data.lower.power,
                data.lower.mean_eff,
                data.lower.type_m,
                data.lower.type_s,
            ],
        }
    )
    return (
        ggplot(df, aes(x="metric", y="value", fill="group"))
        + geom_col(position=position_dodge(width=0.7), width=0.6, color="black")
        + geom_text(
            aes(label="value.round(4)"),
            position=position_dodge(width=0.7),
            va="bottom",
            size=9,
            format_string="{:.4f}",
        )
        + scale_fill_manual(values=["#1f77b4", "#ff7f0e"])
        + labs(x="Metric", y="Value")
        + theme_minimal()
        + theme(
            plot_title=element_text(size=14, face="bold", ha="center"),
            axis_title=element_text(size=12, face="bold"),
            axis_text=element_text(size=10),
            legend_title=element_blank(),
            legend_position="top",
            figure_size=(10, 6),
        )
    )
