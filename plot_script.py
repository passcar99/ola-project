import json
import matplotlib.pyplot as plt
import numpy as np
from environment.Algorithms import regret_bound, clairvoyant
from environment.RandomEnvironment import RandomEnvironment
from environment.Environment import Environment




if __name__ == "__main__":
    filename = "backup/Step4-2022-07-16 15:18:46.202288/GPUCB_Learner4-1"
    data =0
    with open(filename) as fp:
        data = json.load(fp)
    opt = 71903.1402
    negative_regret_traces = []
    positive_regret_traces = []
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

    env = Environment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    optimal_arm, opt = clairvoyant(env, arms, bounds, 300)
    """ rews = []
    for _ in range(10000):
        rews.append(env.round(optimal_arm)[0]['profit'])
    print(np.std(rews))
    plt.title("Reward distribution")
    plt.hist(rews, bins=100)
    plt.show() """
    #opt = np.mean(rews)
    for i, trace in enumerate(data):
        cumsum = np.cumsum(opt-np.array(trace))
        if cumsum[-1] < 0:
            negative_regret_traces.append(cumsum)
            print(np.sum(np.array(trace)>=opt))
            plt.hist(trace, bins = 25, density=True)
        else:
            positive_regret_traces.append(cumsum)
        plt.plot(cumsum)
    plt.show()
    plt.axvline(opt)
    #plt.plot(np.arange(0, 100), [opt]*100)
    
    plt.show()
    for trace in negative_regret_traces:
        plt.plot(np.array(trace))
    plt.show()

    mean = np.mean(positive_regret_traces,axis=0)
    variance = np.var(data)
    regret = regret_bound(1600, variance, 100)
    t = np.arange(1, 101)
    plt.plot(mean/regret)
    plt.show()