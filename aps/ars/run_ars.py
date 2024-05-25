import torch
import gymnasium as gym
from .ars_policy import ARSPolicy
from utils.normalizer import Normalizer
from utils.hyperparameters import ARSHyperparameters


def explore(env: gym.Env, hyperparameters: ARSHyperparameters, normalizer: Normalizer, 
            policy: ARSPolicy, direction=None, delta=None, seed=None) -> float:
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
        state = torch.tensor(state, dtype=torch.float32)
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, truncated, info = env.step(action.numpy())
        sum_rewards += reward
        num_plays += 1

    return sum_rewards


def train(env: gym.Env, policy: ARSPolicy, normalizer: Normalizer, 
          hp: ARSHyperparameters) -> ARSPolicy:
    """
    Learn a policy using augmented random search.

    Args:
        env: A Farama environment that uses scalar states.
        ARSPolicy: An ARS ARSPolicy to train.
        normalizer: A normalizer instance to normalize states.
        hp: Hyperparameters for the model.

    Returns:
        A trained ARSPolicy.
    """
    for step in range(hp.num_experiments):
        deltas = policy.sample_deltas()
        positive_rewards, negative_rewards = [0] * hp.num_rollouts, [0] * hp.num_rollouts
        
        for k in range(hp.num_rollouts):
            positive_rewards[k] = explore(env, hp, normalizer, policy, direction="positive", 
                                          delta=deltas[k])
        
        for k in range(hp.num_rollouts):
            negative_rewards[k] = explore(env, hp, normalizer, policy, direction="negative", 
                                          delta=deltas[k])
        
        all_rewards = torch.tensor(positive_rewards + negative_rewards, dtype=torch.float32)
        sigma_r = all_rewards.std()
        
        # Sorting the best_rewards by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) 
                  in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:hp.b]
        best_rewards = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        policy.update(best_rewards, sigma_r)
        print('Experiment:', step + 1)

    return policy


def run_ars(seeds: list[int], alpha:float=0.02, v: float=0.03, num_rollouts: int=100, 
            b:int=8, num_experiments:int=100, horizon: int=100, 
            rollout_horizon:int = 1000) -> list[float]:
    """
    Run experiments using augmented random search.

    Args:
        seeds: A list of integers to use as seeds for the experiments.
        alpha: The learning rate for ARS.
        v: The amount of noise to perturb the parameters by
        num_rollouts: The number of directions sampled per iteration.
        b: The number of top-performing directions to use to update the parameters of the 
        linear ARSPolicy.
        num_experiments: The maximum number of experiments to try.
        horizon: The maximum number of rollouts per experiment.

    Returns:
        A list of cumulative rewards from deploying each ARSPolicy.
    """
    rewards_list = []

    for s, e in zip(seeds, ("Swimmer-v4", "Hopper-v4", "HalfCheetah-v4", "Walker2d-v4")):
        hp = ARSHyperparameters(alpha, v, num_rollouts, b, num_experiments, horizon)
        env = gym.make(e, render_mode="rgb_array")
        env.reset(seed=s)
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        policy = ARSPolicy(hp, num_inputs, num_outputs)
        normalizer = Normalizer(num_inputs)
        trained_policy = train(env, policy, normalizer, hp)

        # Setting the horizon for evaluation
        hp.horizon = rollout_horizon

        # Evaluating the learned ARSPolicy
        rewards = explore(env, hp, normalizer, trained_policy)
        rewards_list.append(rewards)

    return rewards_list
