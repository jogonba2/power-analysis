class Task:
    def fit(self, X, y=None):
        """
        Fit the task with the provided dataset.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_power(self):
        ...