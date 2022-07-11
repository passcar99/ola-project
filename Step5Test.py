from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner5 import GPTS_Learner5
from learners.GPUCB_Learner5 import GPUCB_Learner5
from learners.TSLearner5Topped5D import GPTS_Learner5Topped5D
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations, clairvoyant
from utils import plot_gaussian_process, save_rewards


EXPERIMENT_NAME = "Step5"


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
    tsTOP5D_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []

    n_experiments = 1

    T = 50


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner5(arms,  conpam_matrix, connectivity_matrix!=0.0, prob_buy, avg_sold, margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner5(arms, conpam_matrix, connectivity_matrix!=0.0, prob_buy, avg_sold, margins, bounds ,'fast')
        tsTOP5D_learner = GPTS_Learner5Topped5D(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

        ts_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        tsTOP5D_learner.avg_n_users = 100

        clairvoyant_rewards = []

        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            pulled_arm_ucb = ucb_learner.pull_arm()
            pulled_arm_5D = tsTOP5D_learner.pull_arm()


            reward_ts = env.round(pulled_arm_ts)
            reward_ucb = env.round(pulled_arm_ucb)
            reward_5D = env.round(pulled_arm_5D)

            
            clairvoyant_rewards.append(opt)

            ts_learner.update(pulled_arm_ts, reward_ts[0])
            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            tsTOP5D_learner.update(pulled_arm_5D, reward_5D[0])

            
            plot_gaussian_process(ts_learner)

        print(ts_learner.con_matrix)

        print(connectivity_matrix)
        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        tsTOP5D_rewards_per_experiment.append(tsTOP5D_learner.collected_rewards)


        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)

    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME, ts_learner.NAME, -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner.NAME, -1)

    print(optimal_alloc, opt)
    plt.figure(0)
    plt.ylabel("Regret, step 5")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ts_rewards_per_experiment, axis = 0)), 'r')
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ucb_rewards_per_experiment, axis = 0)), 'g')
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( tsTOP5D_rewards_per_experiment, axis = 0)), 'b')


    plt.legend(["TS", "UCB"])
    plt.show()

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """