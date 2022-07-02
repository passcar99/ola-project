from RandomEnvironment import RandomEnvironment
from TSLearner4 import GPTS_Learner4
from GPUCB_Learner4 import GPUCB_Learner4
from Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from Algorithms import budget_allocations


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
    unfeasible_arms = []
    for p in range(n_products):
        unfeasible_arms.append(np.logical_or(arms <= bounds[p][0], arms >= bounds[p][1]))

    #clairvoyant
    value_matrix = np.zeros((n_products, n_arms))
    alpha_functions = np.array([ fun(arms) for fun in env.alpha_functions()[0]])
    alpha_functions = alpha_functions/100 #total_mass
    expected_margin = np.zeros((n_products))
    for p in range(n_products):
        expected_margin[p] = env.simplified_round(p, n_sim = 10000)
        value_matrix[p, :] = alpha_functions[p, :]* expected_margin[p] *100 # n_users
        value_matrix[p, unfeasible_arms[p]] = -np.inf
    opt = budget_allocations(value_matrix, arms, subtract_budget=True)[1]
    #print(expected_margin)
    print(budget_allocations(value_matrix, arms, subtract_budget=True), expected_margin.flatten())

    ts_rewards_per_experiment = []
    ucb_rewards_per_experiment = []

    clairvoyant_rewards_per_experiment = []

    n_experiments = 1

    T = 200


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner4(arms, conpam_matrix, connectivity_matrix, prob_buy,  margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner4(arms, conpam_matrix, connectivity_matrix, prob_buy,  margins, bounds ,'fast')
        ts_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        clairvoyant_rewards = []
        clairvoyant_ts5d_rewards = []
        clairvoyant_ucb_rewards = []
        print(avg_sold)
        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            pulled_arm_ucb = ucb_learner.pull_arm()

            reward_ts = env.round_step_4(pulled_arm_ts)
            reward_ucb = env.round_step_4(pulled_arm_ucb)

            
            clairvoyant_rewards.append(opt)

            """ n_users  = 100
            for p in range(n_products):
                value_matrix[p, :] = alpha_functions[p, :]* expected_margin[p] *n_users
                value_matrix[p, unfeasible_arms[p]] = -np.inf
            opt = budget_allocations(value_matrix, arms, subtract_budget=True)[1] """
            ts_learner.update(pulled_arm_ts, reward_ts[0])
            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])

        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

        clairvoyant_rewards_per_experiment.append(clairvoyant_rewards)


    print(budget_allocations(value_matrix, arms, subtract_budget=True)[0])
    plt.figure(0)
    plt.ylabel("Regret, step 4")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ts_rewards_per_experiment, axis = 0)), 'r')
    plt.plot(np.arange(0, T), np.cumsum(np.mean(clairvoyant_rewards_per_experiment, axis = 0)-np.mean( ucb_rewards_per_experiment, axis = 0)), 'g')

    plt.legend(["TS", "UCB"])
    plt.show()

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """