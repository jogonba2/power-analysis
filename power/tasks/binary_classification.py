from power.tasks import Task
from itertools import product
import numpy as np

def make_probability_table(
    baseline_acc: float = 0.5,  # accuracy of the baseline model (Model 1/A)
    delta_acc: float = 0.1,  # accuracy difference between Model 1 and Model 2, \theta_B - baseline_acc
    agreement_rate: (
        float | None
    ) = 0.8,  # agreement rate between the two models (Model 1 and Model 2)
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
    if agreement_rate is not None:
        p11 = (agreement_rate + 2 * baseline_acc + delta_acc - 1) / 2
        p10 = baseline_acc - p11
        p01 = baseline_acc + delta_acc - p11
        p00 = agreement_rate - p11
        probs = np.array([p00, p01, p10, p11])
    else:
        acc1 = baseline_acc
        acc2 = baseline_acc + delta_acc
        if acc1 + acc2 <= 1.0:
            # Case A: no overlapping “both correct” mass possible; set p11 = 0
            p11 = 0.0
            p10 = acc1  # Pr(model1 correct, model2 wrong)
            p01 = acc2  # Pr(model1 wrong, model2 correct)
            p00 = 1.0 - acc1 - acc2
        else:
            # Case B: some overlap is forced; set p11 = acc1 + acc2 - 1
            p11 = acc1 + acc2 - 1.0
            p10 = acc1 - p11  # = acc1 - (acc1 + acc2 - 1) = 1 - acc2
            p01 = acc2 - p11  # = acc2 - (acc1 + acc2 - 1) = 1 - acc1
            p00 = 0.0

        probs = np.array([p00, p01, p10, p11], dtype=float)

    if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
        raise ValueError(
            "The provided parameters do not yield a valid probability table."
        )

    table = probs.reshape(2, 2)
    # some more sanity checks
    assert np.isclose(table[1, :].sum(), baseline_acc), "Error!"
    assert np.isclose(table[:, 1].sum(), baseline_acc + delta_acc), "Error!"
    if agreement_rate is not None:
        assert np.isclose(table.diagonal().sum(), agreement_rate), "Error!"

    return table


def make_dataset(
        n_instances: int|list,
        baseline_acc: float | list,
        delta_acc: float | list,
        agreement_rate: float | list | None = None,
        seed: int = 42):

    grid_for_samples = product(
        np.atleast_1d(baseline_acc),
        np.atleast_1d(delta_acc),
        np.atleast_1d(agreement_rate) if agreement_rate is not None else [None],
        np.atleast_1d(n_instances),
    )

    dataset = []

    for baseline, delta, agreement, size in grid_for_samples:
        try:  # skip infeasible combinations
            table = make_probability_table(baseline, delta, agreement)
        except ValueError:
            continue
        dataset.append(
            {
                "size": size,
                "baseline": baseline,
                "delta": delta,
                "agreement": agreement,
                "prob_table": table,
            }
        )

    return dataset


class BinaryClassification(Task):
    def fit(self, X, y=None):
        """
        Fit the task with the provided dataset.
        """

