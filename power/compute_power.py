import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest


# Could Desi review this? The part of z-test is pure GPT
def compute_power(
    prob_table, dataset_size, alpha=0.05, r=40, test_type="mcnemar"
):
    if test_type == "mcnemar" and prob_table[0, 1] == prob_table[1, 0]:
        return 0, 0, 0, 0

    pvals, diffs = [], []
    flat_p = prob_table.reshape(
        4,
    )

    for _ in range(r):
        samp = np.random.multinomial(dataset_size, flat_p).reshape(2, 2)

        if test_type == "mcnemar":
            diff = (samp[0, 1] - samp[1, 0]) / dataset_size
            res = mcnemar(samp, exact=False)
            pval = res.pvalue

        elif test_type == "ztest":
            # Unpaired proportions: success counts and sample sizes
            np.array([samp[0].sum(), samp[1].sum()])
            nobs = np.array(
                [samp[0].sum() + samp[0][1], samp[1].sum() + samp[1][0]]
            )
            # Avoid division by zero or ill-conditioned sampling
            if np.any(nobs == 0):
                continue
            pval = proportions_ztest(
                count=[samp[0][1], samp[1][1]],
                nobs=[samp[0].sum(), samp[1].sum()],
            )[1]
            diff = (samp[0][1] / samp[0].sum()) - (samp[1][1] / samp[1].sum())

        else:
            raise ValueError("Unsupported test_type. Use 'mcnemar' or 'ztest'.")

        pvals.append(pval)
        diffs.append(diff)

    # Ground-truth difference (note: assumes true probs known)
    if test_type == "mcnemar":
        true_diff = prob_table[0, 1] - prob_table[1, 0]
    else:  # ztest
        rate_1 = prob_table[0, 1] / prob_table[0].sum()
        rate_2 = prob_table[1, 1] / prob_table[1].sum()
        true_diff = rate_1 - rate_2

    true_sign = np.sign(true_diff)
    sig = [(d, p) for d, p in zip(diffs, pvals) if p <= alpha]
    power = sum(1 for d, p in sig if np.sign(d) == true_sign) / r
    mean_eff = np.mean(diffs)
    type_m = (
        np.mean([abs(d) / abs(true_diff) for d, _ in sig]) if sig else np.nan
    )
    type_s = (
        np.mean([np.sign(d) != true_sign for d, _ in sig]) if sig else np.nan
    )

    return power, mean_eff, type_m, type_s


def power_bounds(
    baseline_acc,
    delta_acc,
    dataset_size,
    alpha=0.05,
    r=5000,
    test_type="mcnemar",
):
    acc1, acc2 = baseline_acc, baseline_acc + delta_acc
    # Upper‐bound (max agreement)
    p_both_corr = min(acc1, acc2)
    p_diff = abs(acc1 - acc2)
    p_both_inc = 1 - max(acc1, acc2)
    if acc2 > acc1:
        pu_tab = np.array([[p_both_inc, 0.0], [p_diff, p_both_corr]])
    else:
        pu_tab = np.array([[p_both_inc, p_diff], [0.0, p_both_corr]])
    pu, mu, tmu, tsu = compute_power(pu_tab, dataset_size, alpha, r, test_type)

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
    pl, ml, tml, tsl = compute_power(pl_tab, dataset_size, alpha, r, test_type)

    return {
        "upper_power": pu,
        "upper_mean": mu,
        "upper_typeM": tmu,
        "upper_typeS": tsu,
        "lower_power": pl,
        "lower_mean": ml,
        "lower_typeM": tml,
        "lower_typeS": tsl,
    }
