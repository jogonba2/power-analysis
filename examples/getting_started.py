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


power_estimator.fit(simulated_dataset)

# score is what we have in things like benchmarks
print(
    power_estimator.predict_from_score(
        baseline_acc=0.5, delta_acc=0.01, dataset_size=100, agreement=0.052105
    )
)

print(
    power_estimator.predict_mde(
        n_instances=100, baseline=0.6, agreement=0.052105
    )
)


print(power_estimator.landscape_.head())
