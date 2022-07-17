from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner5 import GPTS_Learner5
from learners.GPUCB_Learner5 import GPUCB_Learner5
from learners.TSLearner5Topped5D import GPTS_Learner5Topped5D
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import plot_gaussian_process, save_rewards, plot_and_save_regrets
import datetime
import warnings
warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "Step5"
DISPLAY_FIGURE=False


if __name__ == '__main__':
    connectivity_matrix = np.array([[0.0, 0.9, 0.3, 0.0, 0.0],
                                    [0.5, 0.0, 0.0, 0.8, 0.0],
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
    tsTOP5D_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []

    n_experiments = 30

    T = 160


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner5(arms,  conpam_matrix, connectivity_matrix!=0.0, prob_buy, avg_sold, margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner5(arms, conpam_matrix, connectivity_matrix!=0.0, prob_buy, avg_sold, margins, bounds ,'fast')
        #tsTOP5D_learner = GPTS_Learner5Topped5D(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

        ts_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        #tsTOP5D_learner.avg_n_users = 100

        clairvoyant_rewards = []

        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            #pulled_arm_ucb = ucb_learner.pull_arm()
            #pulled_arm_5D = tsTOP5D_learner.pull_arm()


            reward_ts = env.round(pulled_arm_ts)
            #reward_ucb = env.round(pulled_arm_ucb)
            #reward_5D = env.round(pulled_arm_5D)

            
            clairvoyant_rewards.append(opt)

            ts_learner.update(pulled_arm_ts, reward_ts[0])
            #ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            #tsTOP5D_learner.update(pulled_arm_5D, reward_5D[0])

        print(ts_learner.con_matrix)

        print(connectivity_matrix)
        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        #tsTOP5D_rewards_per_experiment.append(tsTOP5D_learner.collected_rewards)


        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)
        if e%4==0:
            now = '-'+str(datetime.datetime.now())
            save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME, -1)
            save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME, -1)

    now = '-'+str(datetime.datetime.now())
    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME+now, ts_learner.NAME, -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME+now, ucb_learner.NAME, -1)

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