from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner4 import GPTS_Learner4
from learners.GPUCB_Learner4 import GPUCB_Learner4
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import save_rewards, plot_and_save_rewards

EXPERIMENT_NAME = "Step4"
DISPLAY_FIGURE=True



if __name__ == '__main__':
    connectivity_matrix = np.array([[0, 0.2, 0.4, 0.3, 0.1],
                                    [0.5, 0, 0.1, 0.3, 0.1],
                                    [0.3, 0.2, 0, 0.1, 0.4],
                                    [0.13, 0.17, 0.30, 0, 0.4],
                                    [0.16, 0.34, 0.15, 0.25, 0],
                                    ])
    prob_buy = np.array([0.5, 0.2, 0.5, 0.7, 0.7])
    avg_sold = [6,10,5,5,6]
    margins = [30, 20, 30, 40, 50]
    conpam_matrix = [
        {"alpha_params": [(0, 10, 20), (2, 15, 20),(2, 20, 20),(2, 15, 20),(1, 15, 20)], 
        "features":[0, 0], "total_mass":100, "avg_number":100}, 
                    ]
    arms = np.array([0, 5, 10, 15, 20, 25, 30])
    #bounds = np.array([[5, 100],[0, 80],[0, 50],[20, 100],[0, 100]])
    bounds = np.array([[2, 100],[2, 100],[-1, 100],[2, 100],[-1, 100]])

    env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    
    n_products = len(connectivity_matrix)
    n_arms = len(arms)

    optimal_alloc, opt = clairvoyant(env, arms, bounds, 100)
    ts_rewards_per_experiment = []
    ucb_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []

    n_experiments = 2

    T = 5


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner4(arms, conpam_matrix, connectivity_matrix, prob_buy,  margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner4(arms, conpam_matrix, connectivity_matrix, prob_buy,  margins, bounds ,'fast')
        ts_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        clairvoyant_rewards = []

        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            pulled_arm_ucb = ucb_learner.pull_arm()

            reward_ts = env.round(pulled_arm_ts)
            reward_ucb = env.round(pulled_arm_ucb)

            clairvoyant_rewards.append(opt)

            ts_learner.update(pulled_arm_ts, reward_ts[0])
            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])

        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)


    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME, ts_learner.NAME, -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner.NAME, -1)
    print(optimal_alloc, opt)
    
    plot_and_save_rewards([ts_rewards_per_experiment, ucb_rewards_per_experiment],
                        clairvoyant_rewards_per_experiment, ["TS", "UCB"], EXPERIMENT_NAME, T, display_figure=DISPLAY_FIGURE)

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """