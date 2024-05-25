import torch

class GAPSPolicy:
    """Stores and updates parameters for a linear active policy search policy.

    Attributes:
        theta: The weights for the policy.
        discrete: Whether the action space is discrete.
    """
    def __init__(self, num_input: int, num_output: int, discrete: bool=False):
        self.num_input = num_input
        self.num_output = num_output
        self.theta = torch.rand(num_output, num_input, dtype=torch.float64)
        self.discrete = discrete

    def evaluate(self, state: torch.Tensor):
        """
        Predicts an action to take from a given state.

        Args:
            state: A state to predict an action for.

        Returns:
            A predicted action to take.
        """
        if self.discrete:
            return torch.softmax((torch.matmul(self.theta, state)))
        else:
            return torch.matmul(self.theta, state)
        