import pandas as pd
import gymnasium as gym
from ars import run_ars
from aps import run_gaps
import matplotlib.pyplot as plt
from ars.ars_policy import ARSPolicy
from aps.gaps_policy import GAPSPolicy
from utils.normalizer import Normalizer
from utils.hyperparameters import ARSHyperparameters, GAPSHyperparameters

ROLLOUT_HORIZON = 1000

if __name__ == "__main__":
    seeds_list = [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [112, 123, 134, 145, 146], 
                  [8, 6, 7, 5, 3]]
    ars_dict, gaps_dict = dict(), dict()
    
    for i, envir in enumerate(("Swimmer-v4", "Hopper_v4", "HalfCheetah-v4", "Walker2d-v4")):
        ars_dict[envir] = []
        gaps_dict[envir] = []
        for num_experiments in (100, 1000, 10000):
            ars_score, gaps_score = 0, 0
            seeds = seeds_list[i]
            for seed in seeds:
                
                # Running experiments with augmented random search
                ars_hp = ARSHyperparameters(num_experiments=num_experiments)
                env = gym.make(envir)
                env.reset(seed=seed)
                num_inputs = env.observation_space.shape[0]
                num_outputs = env.action_space.shape[0]
                ars_policy = ARSPolicy(ars_hp, num_inputs, num_outputs)
                normalizer = Normalizer(num_inputs)
                trained_ars_policy = run_ars.train(env, ars_policy, normalizer, ars_hp)

                # Setting the horizon for evaluation
                ars_hp.horizon = ROLLOUT_HORIZON

                ars_rewards = run_ars.explore(env, ars_hp, normalizer, trained_ars_policy)
                ars_score += ars_rewards

                # Running the experiments for GAPS
                gaps_hp = GAPSHyperparameters(num_experiments=num_experiments)
                env = gym.make(envir)
                env.reset(seed=seed)
                gaps_policy = GAPSPolicy(num_inputs, num_outputs)
                best_theta = run_gaps.train(env, gaps_policy, normalizer, gaps_hp)
                trained_gaps_policy = GAPSPolicy(num_inputs, num_outputs)
                trained_gaps_policy.theta = best_theta
                gaps_hp.horizon = ROLLOUT_HORIZON
                gaps_rewards = run_gaps.explore(env, gaps_hp, normalizer, 
                                                trained_gaps_policy)
                gaps_score += gaps_rewards

            ars_score /= len(seeds)
            gaps_score /= len(seeds)
            ars_dict[envir].append(ars_score)
            gaps_dict[envir].append(gaps_score)

    # The code below is from ChatGPT because making facet grids with matplotlib is a pain
    data = []

    for algo_name, algo_data in zip(['ARS', 'GAPS'], [ars_dict, gaps_dict]):
        for env, rewards in algo_data.items():
            for num_experiments, reward in zip([100, 1000, 10000], rewards):
                data.append([algo_name, env, num_experiments, reward])

    df = pd.DataFrame(data, columns=['Algorithm', 'Environment', 'Experiments', 'Reward'])

    # Create a facet grid
    algorithms = df['Algorithm'].unique()
    environments = df['Environment'].unique()

    fig, axes = plt.subplots(nrows=len(algorithms), ncols=len(environments), 
                             figsize=(15, 10), sharey=True)
    
    for i, algo in enumerate(algorithms):
        for j, env in enumerate(environments):
            ax = axes[i, j]
            subset = df[(df['Algorithm'] == algo) & (df['Environment'] == env)]
            ax.bar(subset['Experiments'], subset['Reward'])
            if i == 0:
                ax.set_title(env)
            if j == 0:
                ax.set_ylabel(algo)
            if i == len(algorithms) - 1:
                ax.set_xlabel('Experiments')

    plt.tight_layout()
    plt.savefig('aps/results/figure1.png')  # Save the plot as a PNG file
    plt.show()
