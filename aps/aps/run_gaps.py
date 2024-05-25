import torch
import torch
import numpy as np
import gymnasium as gym
from .gaps_policy import GAPSPolicy
from utils.normalizer import Normalizer
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.linear_model import LogisticRegression
from utils.hyperparameters import GAPSHyperparameters
from botorch.models.transforms import Normalize, Standardize


def explore(env: gym.Env, hyperparameters: GAPSHyperparameters, normalizer: Normalizer, 
            policy: GAPSPolicy, seed=None) -> float:
    """
    Roll out a policy in the environment

    Args:
        env: A Farama environment that uses scalar states.
        hyperameters: An instance of the Hyperparameters class.
        normalizer: A normalizer instance to normalize states.
        policy: An instance of the ARSPolicy class.
        direction: The direction of the perturbations, either positive or negative.
        delta: optional perturbations.

    Returns:
        The cumulative rewards gained for a rollout over a finite horizon.
    """
    if seed:
        state, info = env.reset(seed)
    else:
        state, info = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0

    while not done and num_plays < hyperparameters.horizon:
        state = torch.tensor(state, dtype=torch.float64)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state)
        state, reward, done, truncated, info = env.step(action.numpy())
        sum_rewards += reward
        num_plays += 1

    return sum_rewards


def rollout_policies(env: gym.Env, policy: GAPSPolicy, normalizer: Normalizer, 
                     hp: GAPSHyperparameters, points_to_query=None, 
                     seed=None) -> tuple[torch.Tensor]:
    """
    Roll out policies parameterized by given thetas.

    Args:
        env: A Farama environment that uses scalar states.
        policy: An instance of the GAPSPolicy class.
        normalizer: A normalizer instance to normalize states.
        hyperameters: An instance of the GAPSHyperparameters class.
        points_to_query: data to label.

    Returns:
        A tuple with tensors of the points to query and their cumulative rewards.
    """
    if not points_to_query:
        points_to_query = list(
            torch.from_numpy(
                np.random.uniform(
                    low=-0.1, high=0.1, size=policy.theta.shape)).to(torch.float64) 
                    for i in range(hp.num_initial_points))
    
    initial_rewards = []
    
    # Rollout the initial thetas to get an ititial dataset to use
    for theta in points_to_query:
        policy.theta = theta
        if seed:
            reward = explore(env, hp, normalizer, policy)
        else:
            reward = explore(env, hp, normalizer, policy, seed)
        initial_rewards.append(reward)

    # Flatten our 2D representations of thetas and tensor representation of rewards
    points_to_query = torch.stack(points_to_query, dim=0)
    l, w, d = points_to_query.shape
    points_to_query = points_to_query.reshape((l, w * d))
    initial_rewards = torch.tensor(initial_rewards, dtype=torch.float64)

    return points_to_query, initial_rewards.reshape((initial_rewards.shape[0], 1))


def get_best_theta(gp:SingleTaskGP, size: int) -> torch.Tensor:
    """
    Find the parameters that maximize the posterior mean of the cumulative rewards.

    Args:
        gp: A trained SingleTaskGP.
        size: The number of columns in the feature data used to train the GP.

    Returns:
        A tensor containing the best (normalized) parameters.
    """
    bounds = torch.tensor([[-1.0] * size, [1.0] * size], dtype=torch.float64)
    acq_fun = PosteriorMean(gp)

    best_theta, value = optimize_acqf(acq_fun, bounds, 1, 10, 100)

    return best_theta, value


def make_discriminator_data(training_thetas: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Generate data to predict whether it is labeled or not.

    Args:
        training_thetas: A tensor of parameters that have been labeled.

    Returns:
        A tuple with tensors for training features, training targets, and prediction 
        features from the u nlabeled data.
    """
    l, w = training_thetas.shape

    # Since we don't have a fixed dataset, we need to sample from the range of parameters
    # Our training set will be half positive and half negative examples to avoid issues 
    # with imbalance
    candidate_points = torch.from_numpy(
        np.random.uniform(low=-0.75, high=0.75, size=(l, w))
        ).to(torch.float64)
        
    negative_x, negative_y = candidate_points, torch.zeros(candidate_points.shape[0], 
                                                               dtype=torch.float64)
        
    positive_x, positive_y = training_thetas, torch.ones(l, dtype=torch.float64)
        
    discriminator_x = torch.vstack([negative_x, positive_x])
    discriminator_y = torch.cat([negative_y, positive_y])

    return discriminator_x, discriminator_y, negative_x


def fit_gp(x: torch.Tensor, y: torch.Tensor, env: gym.Env) -> SingleTaskGP:
    """
    Fit a Gaussian process regression.

    Args:
        x: A tensor of features.
        y: A tensor of targets to predict.

    Returns:
        A trained SingleTaskGP.

    """
    gp = SingleTaskGP(x, y, 
                      input_transform=Normalize(d=x.shape[-1]), 
                      outcome_transform=Standardize(y.shape[-1]))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return gp


def train(env: gym.Env, policy: GAPSPolicy, normalizer: Normalizer, hp: GAPSHyperparameters, 
          seed: int, num_top_samples: int=1):
    """
    Train a linear policy using GAPS.

    Args:
        env: A gymnasium environment.
        policy: An instance of GAPSPolicy
        normalizer: An instance of the Normalizer class.
        hp: An instance of the GAPSHyperparameters class.
        num_top_samples: the size of the batch to sue in the acquisition function.

    Returns:
        A tensor with the best performing parameters.
    """
    
    # Sample some initial points
    training_thetas, training_rewards = rollout_policies(env, policy, normalizer, hp, 
                                                         seed=seed)

    # Train a GP on the initial points. Note we don't have to use a GP here.
    gp = fit_gp(training_thetas, training_rewards, env)

    for experiment in range(hp.num_experiments):
        discriminator_x, discriminator_y, neg_x = make_discriminator_data(training_thetas)

        # Predict if a point is in the labeled dataset or not
        clf = LogisticRegression().fit(discriminator_x, discriminator_y)
        cls_probs = clf.predict_proba(neg_x)

        # Get the points with the highest predictions for being unlabeled
        idx = torch.tensor(cls_probs, dtype=torch.float64)[:, 0].topk(num_top_samples)[1]
        points_to_label = neg_x[idx, :]

        # Reshaping thetas to num_points x num_actions x num_states
        points_to_label = list(points_to_label.reshape((points_to_label.shape[0], 
                                                        policy.num_output, 
                                                        policy.num_input)))
        
        labeled_thetas, labeled_rewards = rollout_policies(env, policy, normalizer, hp, 
                                                           points_to_label)
        
        # Update the training dataset for the GP
        training_thetas = torch.vstack((training_thetas, labeled_thetas))
        training_rewards = torch.cat((training_rewards, labeled_rewards))

        # Train the model with the new data
        gp = fit_gp(training_thetas, training_rewards, env)

        print("Experiment: " + str(experiment + 1))

    best_theta, value = get_best_theta(gp, training_thetas.shape[1])

    return best_theta.reshape(policy.theta.shape), value
 
