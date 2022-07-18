import json
import matplotlib.pyplot as plt
import numpy as np
from environment.Algorithms import regret_bound, clairvoyant
from environment.RandomEnvironment import RandomEnvironment
from environment.Environment import Environment




if __name__ == "__main__":
    filenames = ["backup/WHOLE_REWARDS/backup/WHOLE_STEP3/TS-1"]#, "backup/WHOLE_REWARDS/backup/WHOLE_STEP5/UCB-1"]
    names = ["TS", "UCB"]
    colors = ['r', 'g']
    data =[]
    for filename in filenames:
        with open(filename) as fp:
            data.append(json.load(fp))
    opt = 72176.05288485 # fully
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
        {"alpha_params": [(10, 50, 10), (5, 30, 15),(0, 50, 100),(0, 50, 100),(0, 40, 100)], #Private Rich
        "features":0, "total_mass":100, "avg_number":25}, 
        {"alpha_params": [(15, 50, 10), (10, 30, 15),(0, 50, 50),(2, 50, 50),(1, 40, 50)], #Private Poor
        "features":1, "total_mass":100, "avg_number":25},
        {"alpha_params": [(0, 30, 100), (0, 50, 50),(5, 20, 7),(8, 20, 10),(10, 25, 5)], #Company Rich
        "features":2, "total_mass":100, "avg_number":25},
        {"alpha_params": [(0, 30, 100), (0, 50, 50),(5, 40, 7),(8, 40, 10),(10, 50, 5)], #Company Poor
        "features":3, "total_mass":100, "avg_number":25}]
    arms = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    bounds = np.array([[-1, 100],[-2, 100],[-1, 100],[-1, 100],[-1, 100]])
    

    
    for i, learner_traces in enumerate(data):
        negative_regret_traces.append([])
        positive_regret_traces.append([])
        for trace in learner_traces:
            cumsum = np.cumsum(opt-np.array(trace))
            if cumsum[-1] < 0:
                negative_regret_traces[-1].append(cumsum)
                print(np.sum(np.array(trace)>=opt))
                plt.hist(trace, bins = 25, density=True)
            else:
                positive_regret_traces[-1].append(cumsum)
    
    plt.show()
    for learner_traces in negative_regret_traces:
        for trace in learner_traces:
            plt.plot(np.array(trace))
    plt.show()

    """ from scipy.optimize import minimize, LinearConstraint
    fun = lambda x: np.square(x - 2*(1-np.exp(-x)))
    x = minimize(fun, 2, constraints=[LinearConstraint(1, 0, 1000)]).x
    print(minimize(fun, 5))
    print(x/(1-np.exp(-x))) """
    mean_ts = np.mean(positive_regret_traces[0],axis=0)
    variance_ts = np.var(data[0])
    t = np.arange(1, len(mean_ts))
    regret = regret_bound(1672.0243094153861, 0.1, len(mean_ts))
    """ mean_ucb = np.mean(positive_regret_traces[1],axis=0)
    variance_ucb = np.var(data[1]) """
    t = np.arange(1, len(mean_ts))
    plt.plot(mean_ts/regret, color='r')
    #plt.plot(mean_ucb/regret, color='g')
    plt.legend(["TS", "UCB"])
    plt.show()