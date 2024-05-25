import torch

class Normalizer:
    """Normalizes states while training and implementing a policy.

    Attributes:
        n: The number of states that have been observed.
        mean: The mean state for each rollout.
        mean_diff: The average difference between a state and the average state.
        var: The variance of states.
    """

    def __init__(self, num_inputs:int) -> None:
        """Initializes a normalizer for ARS training.

        Args:
            num_inputs: The number of states to be normalized over.
        """
        self.n = 0.0
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, X: torch.Tensor) -> None:
        """Updates the internal variables after observing a new state.

        Args:
            X: A newly observed state.
        """
        self.n += 1.0
        last_mean = self.mean.clone()
        self.mean.add_((X - self.mean) / self.n)
        self.mean_diff.add_((X - last_mean) * (X - self.mean))
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalizes a newly observed state.

        Args:
            inputs: A newly observed state.

        Returns:
            A standardized state.
        """
        obs_mean = self.mean
        obs_std = torch.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
