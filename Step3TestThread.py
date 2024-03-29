from fileinput import filename
from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner import GPTS_Learner
from learners.TSLearnerTopped5D import GPTS_Learner_TOP5D
from learners.GPUCB_Learner import GPUCB_Learner
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import plot_gaussian_process, save_rewards, plot_and_save_regrets
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "Step3"
DISPLAY_FIGURE=True


def experiment(e, pbar=None):
    env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    ts_learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
    ucb_learner = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
    ts_learner.avg_n_users = 100
    ucb_learner.avg_n_users = 100
    clairvoyant_rewards = [opt]*T
    print("Experiment: " + str(e))
    for t in tqdm(range(0, T)):
        pulled_arm_ts = ts_learner.pull_arm()
            #pulled_arm_5D = tsTOP5D_learner.pull_arm()
        pulled_arm_ucb = ucb_learner.pull_arm()
            #pulled_arm_ucb_paper = ucb_learner_paper.pull_arm()

        reward_ts = env.round(pulled_arm_ts)
            #reward_5D = env.round(pulled_arm_5D)
        reward_ucb = env.round(pulled_arm_ucb)
            #reward_ucb_paper = env.round(pulled_arm_ucb_paper)
            
        ts_learner.update(pulled_arm_ts, reward_ts[0])
            #tsTOP5D_learner.update(pulled_arm_5D, reward_5D[0])
        ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            #ucb_learner_paper.update(pulled_arm_ucb_paper, reward_ucb_paper[0])

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        #ucb_paper_rewards_per_experiment.append(ucb_learner_paper.collected_rewards)
        #tsTOP5D_rewards_per_experiment.append(tsTOP5D_learner.collected_rewards)

    clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)
    #pbar.update(1)

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
    unfeasible_arms = []
    for p in range(n_products):
        unfeasible_arms.append(np.logical_or(arms <= bounds[p][0], arms >= bounds[p][1]))

    #clairvoyant
    
    #print(expected_margin)
    optimal_alloc, opt = clairvoyant(env, arms, bounds, total_mass=100)
    print(optimal_alloc, opt)

    ts_rewards_per_experiment = []
    ucb_rewards_per_experiment = []
    #ucb_paper_rewards_per_experiment = []
    tsTOP5D_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []
    n_experiments = 3

    T = 50


    with tqdm(range(0, n_experiments)) as pbar:
        with ProcessPoolExecutor(max_workers=4) as ex:
            for e in range(0, n_experiments):
                ex.submit(experiment(e))

    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME, "TSLearner", -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME, "UCBLearner", -1)
    print(optimal_alloc, opt)
    plot_and_save_regrets([ts_rewards_per_experiment, 
                            #tsTOP5D_rewards_per_experiment, 
                            ucb_rewards_per_experiment],
                        clairvoyant_rewards_per_experiment, ["TS", 
                        #"TSTOP5D", 
                        "UCB"], EXPERIMENT_NAME, T, display_figure=DISPLAY_FIGURE)

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """