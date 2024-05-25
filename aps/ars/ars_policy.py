import torch
from utils.hyperparameters import ARSHyperparameters

class ARSPolicy:
    """
    Stores and updates parameters for a linear augmented random search policy.

    Attributes:
        theta: The weights for the linear policy.
        hp: ARS for learning a linear ARSPolicy.
    """
    
    def __init__(self, hp: ARSHyperparameters, input_size: int, output_size: int) -> None:
        """
        Initializes a linear ARSPolicy for augmented random search.

        Args:
            ARS: A ARS object with ARS for training a 
                linear ARSPolicy.
            input_size: The size of observations from the environment.
            output_size: The size of the actions space for the environment.
        """
        self.theta = torch.zeros(output_size, input_size, dtype=torch.float)
        self.hp = hp
    
    def evaluate(self, state: torch.Tensor, delta=None, direction=None) -> torch.Tensor:
        """
        Predicts an action to take from a given state.

        Args:
            state: A state to predict an action for.
            delta: Samples from a standard normal distribution.
            direction: The direction the original ARSPolicy should be perturbed.

        Returns:
            A predicted action to take.
        """
        if direction == "positive":
            return torch.matmul(self.theta + delta, state)
        elif direction == "negative":
            return torch.matmul(self.theta - delta, state)
        else:
            return torch.matmul(self.theta, state)
    
    def sample_deltas(self) -> list:
        """
        Samples perturbations from a standard normal distribution.

        Returns:
            A list of perturbations sampled from a random normal distribution.
        """
        return [torch.randn_like(self.theta) for _ in range(self.hp.num_rollouts)]
    
    def update(self, best_rewards: torch.Tensor, sigma_r: float) -> None:
        """Updates the parameters for the linear ARSPolicy.

        Args:
            best_rewards: A list of the highest rewards from the rollouts for an update.
            sigma_r: The standard deviation of the rewards.
        """
        step = torch.zeros_like(self.theta)
        for r_pos, r_neg, d in best_rewards:
            step.add_((r_pos - r_neg) * d)
            
        self.theta.add_(self.hp.alpha / (self.hp.num_rollouts * sigma_r) * step)
