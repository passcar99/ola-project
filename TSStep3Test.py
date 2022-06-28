from RandomEnvironment import RandomEnvironment
from TSLearner import GPTS_Learner
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
    avg_sold = [3,2,3,1,2]
    margins = [30, 20, 30, 40, 50]
    conpam_matrix = [
        {"alpha_params": [(0, 20, 10), (4, 20, 20),(4, 20, 10),(4, 15, 12),(4, 20, 15)], 
        "features":[0, 0], "total_mass":64, "avg_number":100}, 
                    ]
    arms = np.array([5, 10, 15, 20])
    #bounds = np.array([[5, 100],[0, 80],[0, 50],[20, 100],[0, 100]])
    bounds = np.array([[0, 100],[0, 100],[0, 100],[0, 100],[0, 100]])

    env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    
    n_products = len(connectivity_matrix)
    n_arms = len(arms)
    unfeasible_arms = []
    for p in range(n_products):
        unfeasible_arms.append(np.logical_or(arms <= bounds[p][0], arms >= bounds[p][1]))

    #clairvoyant
    value_matrix = np.zeros((n_products, n_arms))
    alpha_functions = np.array([fun(arms) for fun in env.alpha_functions()[0]])
    alpha_functions = alpha_functions/alpha_functions.sum(axis=1).reshape(-1, 1)
    for p in range(n_products):
        expected_margin = env.simplified_round(p, n_sim = 1000)
        value_matrix[p, :] = alpha_functions[p, :]* expected_margin *100 # n_users
        value_matrix[p, unfeasible_arms[p]] = -np.inf
    opt = budget_allocations(value_matrix, arms, subtract_budget=True)[1]

    
    ts_rewards_per_experiment = []
    n_experiments = 1

    T = 150


    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

        for t in tqdm(range(0, T)):
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward[0])

        print(ts_learner.collected_rewards, opt)
        ts_rewards_per_experiment.append(ts_learner.collected_rewards)

    print(budget_allocations(value_matrix, arms, subtract_budget=True)[0])
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.cumsum(opt-np.mean( ts_rewards_per_experiment, axis = 0)), 'r')
    #plt.plot(np.arange(0, T),np.mean( ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["TS", "UCB"])
    plt.show()