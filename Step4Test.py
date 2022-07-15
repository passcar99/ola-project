from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner4 import GPTS_Learner4
from learners.GPUCB_Learner4 import GPUCB_Learner4
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import save_rewards, plot_and_save_rewards
import datetime

EXPERIMENT_NAME = "Step4"
DISPLAY_FIGURE=False



if __name__ == '__main__':
    connectivity_matrix = np.array([[0, 0.9, 0.3, 0.0, 0.0],
                                    [0.5, 0, 0, 0.8, 0],
                                    [0.0, 0.0, 0.0, 0.6, 0.6],
                                    [0.0, 0.0, 0.7, 0.0, 0.9],
                                    [0.0, 0.0, 0.7, 0.9, 0],
                                    ])
    prob_buy = np.array([0.8, 0.5, 0.9, 0.7, 0.3])
    #prob_buy = np.array([1, 1, 1, 1, 1])
    avg_sold = [2,4,1.5,2,3]
    margins = [1000, 300, 100, 75, 30]
    conpam_matrix = [
        {"alpha_params": [(0, 30, 50*3), (0, 25, 5*3),(5, 20, 10*3),(5, 40, 15*3),(5, 25, 20*3)], 
        "features":[0, 0], "total_mass":300, "avg_number":100}, 
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

    n_experiments = 10

    T = 365


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
        if e%4==0:
            now = '-'+str(datetime.datetime.now())
            save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME, -1)
            save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME, -1)

    now = '-'+str(datetime.datetime.now())
    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME, -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME, -1)
    
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