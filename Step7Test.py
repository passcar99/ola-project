from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner5 import GPTS_Learner5
from learners.GPUCB_Learner5 import GPUCB_Learner5
from learners.TSLearner5Topped5D import GPTS_Learner5Topped5D
from learners.ContextManager import ContextManager
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import plot_gaussian_process, save_rewards, plot_and_save_regrets
import warnings
import datetime
warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "Step7"
DISPLAY_FIGURE=True


if __name__ == '__main__':
    connectivity_matrix = np.array([[0, 0.9, 0.3, 0.0, 0.0],
                                    [0.5, 0, 0, 0.8, 0],
                                    [0.0, 0.0, 0.0, 0.6, 0.6],
                                    [0.0, 0.0, 0.7, 0.0, 0.9],
                                    [0.0, 0.0, 0.7, 0.9, 0],
                                    ])
    prob_buy = np.array([0.8, 0.5, 0.9, 0.7, 0.3])
    avg_sold = [2,4,1.5,2,3]
    margins = [1000, 300, 100, 75, 30]
    conpam_matrix = [ 
        {"alpha_params": [(0, 15, 10), (0, 20, 80),(5, 50, 50),(0, 50, 100),(0, 30, 100)], #Private Rich
        "features":0, "total_mass":100, "avg_number":25}, 
        {"alpha_params": [(0, 10, 10), (0, 10, 20),(5, 50, 20),(2, 50, 50),(1, 20, 50)], #Private Poor
        "features":1, "total_mass":100, "avg_number":25},
        {"alpha_params": [(0, 20, 80), (0, 20, 30),(5, 20, 7),(8, 20, 10),(10, 25, 5)], #Company Rich
        "features":2, "total_mass":100, "avg_number":25},
        {"alpha_params": [(0, 20, 80), (0, 20, 30),(5, 40, 7),(8, 40, 10),(10, 25, 5)], #Company Poor
        "features":3, "total_mass":100, "avg_number":25
        }]
    arms = np.array([0,5, 10, 15, 20, 25,30, 35, 40])
    bounds = np.array([[-1, 15],[-1, 100],[1, 100],[-1, 100],[-1, 100]])

    env = Environment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    
    n_products = len(connectivity_matrix)
    n_arms = len(arms)
    optimal_alloc, opt = clairvoyant(env, arms, bounds, 100, class_mask=[0, 1, 2, 2]) # last two classes are the same
    print("OPTIMAL ALLOCATION AND VALUE--")
    print(optimal_alloc, opt)
    print("------------------------------")
    optimal_alloc, opt = clairvoyant(env, arms, bounds, 100, ) # last two classes are the same
    print("OPTIMAL ALLOCATION AND VALUE--")
    print(optimal_alloc, opt)
    print("------------------------------")
    ts_rewards_per_experiment = []
    ucb_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []

    n_experiments = 1

    T = 100


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = ContextManager(arms,  conpam_matrix, connectivity_matrix, prob_buy,  margins, bounds ,'fast', "TS")
        ucb_learner = ContextManager(arms, conpam_matrix, connectivity_matrix, prob_buy, margins, bounds ,'fast', "UCB")

        ts_learner.avg_n_users = 100
        #ucb_learner.avg_n_users = 100

        clairvoyant_rewards = []

        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            print(pulled_arm_ts)
            #pulled_arm_ucb = ucb_learner.pull_arm()
            #print(pulled_arm_ucb)


            reward_ts = env.round(pulled_arm_ts, observed_features=True)
            #reward_ucb = env.round(pulled_arm_ucb, observed_features=True)
            print("--", np.sum(r['profit'] for r in reward_ts), opt)
            
            clairvoyant_rewards.append(opt)

            ts_learner.update(pulled_arm_ts, reward_ts)
            #ucb_learner.update(pulled_arm_ucb, reward_ucb)

            
        #print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        #ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        if e%4==0:
            now = '-'+str(datetime.datetime.now())
            save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME+"TS", -1)
            save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME+"UCB", -1)


        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)

    now = '-'+str(datetime.datetime.now())
    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME+"TS", -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME+"UCB", -1)

    #print(optimal_alloc, opt)

    plot_and_save_regrets([ts_rewards_per_experiment, #ucb_rewards_per_experiment
    ],
                        clairvoyant_rewards_per_experiment, ["TS",  "UCB"], EXPERIMENT_NAME, T, display_figure=DISPLAY_FIGURE)

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """