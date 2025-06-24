import numpy as np

from power.tasks.binary_classification import BinaryClassification, make_dataset

# create dataset using binary_classification.make_dataset()
simulated_dataset = make_dataset(
    n_instances=[10, 20, 50, 100, 500],
    baseline_acc=np.linspace(0.5, 0.9, 20, dtype=np.float32),
    delta_acc=np.linspace(0.01, 0.2, 20, dtype=np.float32),
    agreement_rate=np.linspace(0.0, 0.99, 20, dtype=np.float32),
    seed=1,
)

power_estimator = BinaryClassification(
    n_iterations=50, random_state=42, n_jobs=-1
)


# WIP
power_estimator.predict_from_score(
    baseline_acc=0.8, delta_acc=0.1, dataset_size=47, agreement=0.5
)

power_estimator.fit(simulated_dataset)

print(power_estimator.landscape_)
