from RandomEnvironment import RandomEnvironment
from TSLearner import GPTS_Learner
from Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    connectivity_matrix = np.array([[0, 0.2, 0.4, 0.3, 0.1],
                                    [0.5, 0, 0.1, 0.3, 0.1],
                                    [0.3, 0.2, 0, 0.1, 0.4],
                                    [0.13, 0.17, 0.30, 0, 0.4],
                                    [0.16, 0.34, 0.15, 0.25, 0],
                                    ])
    prob_buy = np.array([0.1, 0.2, 0.5, 0.9, 0.7])
    avg_sold = [5,6,7,8,9]
    margins = [10, 20, 30, 40, 50]
    conpam_matrix = [{"alpha_params": [(0, 10, 2), (5, 10, 6),(5, 20, 10),(5, 50, 6),(5, 8, 6)], "features":[0, 0], "total_mass":64, "avg_number":100}, 
                    ]
    arms = np.array([20, 30, 40, 50, 60])
    bounds = np.array([[5, 100],[0, 80],[0, 50],[20, 100],[0, 100]])


    ts_rewards_per_experiment = []

    n_experiments = 1

    T = 50


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)

        ts_learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

        for t in tqdm(range(0, T)):
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward[0]['alphas'])

        ts_rewards_per_experiment.append(ts_learner.collected_rewards)


    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean( ts_rewards_per_experiment, axis = 0)), 'r') # TODO use regret instead
    plt.legend(["TS", "UCB"])
    plt.show()