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
from utils import plot_gaussian_process, save_rewards

EXPERIMENT_NAME = "Step3"
DISPLAY_FIGURE=True

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
        {"alpha_params": [(20, 30, 20), (2, 15, 20),(2, 20, 20),(2, 15, 20),(1, 15, 20)], 
        "features":[0, 0], "total_mass":100, "avg_number":100}, 
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
    n_experiments = 2

    T = 10


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        tsTOP5D_learner = GPTS_Learner_TOP5D(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        #ucb_learner_paper = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast', 'paper')
        ts_learner.avg_n_users = 100
        tsTOP5D_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        clairvoyant_rewards = [opt]*T
        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            pulled_arm_5D = tsTOP5D_learner.pull_arm()
            pulled_arm_ucb = ucb_learner.pull_arm()
            #pulled_arm_ucb_paper = ucb_learner_paper.pull_arm()

            reward_ts = env.round(pulled_arm_ts)
            reward_5D = env.round(pulled_arm_5D)
            reward_ucb = env.round(pulled_arm_ucb)
            #reward_ucb_paper = env.round(pulled_arm_ucb_paper)
            
            ts_learner.update(pulled_arm_ts, reward_ts[0])
            tsTOP5D_learner.update(pulled_arm_5D, reward_5D[0])
            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            #ucb_learner_paper.update(pulled_arm_ucb_paper, reward_ucb_paper[0])
            
            plot_gaussian_process(ts_learner)

        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        #ucb_paper_rewards_per_experiment.append(ucb_learner_paper.collected_rewards)
        tsTOP5D_rewards_per_experiment.append(tsTOP5D_learner.collected_rewards)

        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)

    save_rewards(ts_rewards_per_experiment, EXPERIMENT_NAME, ts_learner.NAME, -1)
    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner.NAME, -1)
    print(optimal_alloc, opt)
    fig = plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ts_rewards_per_experiment, axis = 0)), 'r')
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( tsTOP5D_rewards_per_experiment, axis = 0)), 'b')
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ucb_rewards_per_experiment, axis = 0)), 'g')

    plt.legend(["TS", "TSTOP5D", "UCB"])
    
    file_name = 'backup/'+EXPERIMENT_NAME+'_rewards.png'
    plt.savefig(fname=file_name)
    if DISPLAY_FIGURE:
        plt.show()
    else:
        plt.close(fig)

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """