import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest

from .logging import get_logger, log
from .types import PowerBounds, PowerOutput

_logger = get_logger(__name__)


def compute_power(
    prob_table: np.ndarray,
    dataset_size: int,
    alpha: float = 0.05,
    r: int = 40,
    test_type: str = "mcnemar",
    seed: int = 13,
) -> PowerOutput:
    if test_type == "mcnemar" and prob_table[0, 1] == prob_table[1, 0]:
        log(
            _logger.info,
            "The null hypothesis holds in McNemar's test for this sample; "
            "since there is no difference between the two models, the "
            "statistical power will be 0.",
            "yellow",
        )
        return PowerOutput(power=0.0, mean_eff=0.0, type_m=0.0, type_s=0.0)

    pvals, diffs = [], []
    flat_p = prob_table.reshape(
        4,
    )

    rng = np.random.default_rng(seed)
    for _ in range(r):
        samp = rng.multinomial(dataset_size, flat_p).reshape(2, 2)

        if test_type == "mcnemar":
            diff = (samp[0, 1] - samp[1, 0]) / dataset_size
            res = mcnemar(samp, exact=False)
            pval = res.pvalue

        elif test_type == "ztest":
            n1, n2 = samp[0].sum(), samp[1].sum()
            if n1 == 0 or n2 == 0:
                continue  # skip iteration due to invalid sample

            # Proportion of "successes"
            prop1 = samp[0][1] / n1
            prop2 = samp[1][1] / n2
            diff = prop1 - prop2

            pval = proportions_ztest(
                count=[samp[0][1], samp[1][1]], nobs=[n1, n2]
            )[1]

        else:
            raise ValueError("Unsupported test_type. Use 'mcnemar' or 'ztest'.")

        pvals.append(pval)
        diffs.append(diff)

    # Compute true effect size
    if test_type == "mcnemar":
        true_diff = prob_table[0, 1] - prob_table[1, 0]
    else:  # ztest
        sum0, sum1 = prob_table[0].sum(), prob_table[1].sum()
        if sum0 == 0 or sum1 == 0:
            true_diff = np.nan
        else:
            rate_1 = prob_table[0, 1] / sum0
            rate_2 = prob_table[1, 1] / sum1
            true_diff = rate_1 - rate_2

    true_sign = np.sign(true_diff) if not np.isnan(true_diff) else 0
    sig = [(d, p) for d, p in zip(diffs, pvals) if p <= alpha]
    power = (
        sum(1 for d, _ in sig if np.sign(d) == true_sign) / r
        if r > 0
        else np.nan
    )
    mean_eff = np.mean(diffs) if diffs else np.nan
    type_m = (
        np.mean([abs(d) / abs(true_diff) for d, _ in sig])
        if sig and true_diff != 0
        else np.nan
    )
    type_s = (
        np.mean([np.sign(d) != true_sign for d, _ in sig]) if sig else np.nan
    )

    return PowerOutput(
        power=power, mean_eff=mean_eff, type_m=type_m, type_s=type_s
    )


def power_bounds(
    baseline_acc: float,
    delta_acc: float,
    dataset_size: int,
    alpha: float = 0.05,
    r: int = 5000,
    test_type: str = "mcnemar",
) -> PowerBounds:
    acc1, acc2 = baseline_acc, baseline_acc + delta_acc
    # Upper‐bound (max agreement)
    p_both_corr = min(acc1, acc2)
    p_diff = abs(acc1 - acc2)
    p_both_inc = 1 - max(acc1, acc2)
    if acc2 > acc1:
        pu_tab = np.array([[p_both_inc, 0.0], [p_diff, p_both_corr]])
    else:
        pu_tab = np.array([[p_both_inc, p_diff], [0.0, p_both_corr]])
    upper_bound = compute_power(pu_tab, dataset_size, alpha, r, test_type)

    # Lower‐bound (max disagreement)
    if (2 - acc1 - acc2) <= 1:
        p_neither = 0.0
        only1 = 1 - acc1
        only2 = 1 - acc2
        p_both = 1 - only1 - only2
    else:
        p_both = 0.0
        only1 = acc1
        only2 = acc2
        p_neither = 1 - only1 - only2
    pl_tab = np.array([[p_neither, only1], [only2, p_both]])
    lower_bound = compute_power(pl_tab, dataset_size, alpha, r, test_type)

    return PowerBounds(upper=upper_bound, lower=lower_bound)


if __name__ == "__main__":
    a = power_bounds(0.5, 0.2, 1000, alpha=0.05, r=500, test_type="mcnemar")
