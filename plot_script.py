import json
import matplotlib.pyplot as plt
import numpy as np
from environment.Algorithms import regret_bound




if __name__ == "__main__":
    filename = "backup/Step4-2022-07-16 13:41:06.584661/GPTS_Learner4-1"
    data =0
    with open(filename) as fp:
        data = json.load(fp)
    opt = 71903.1402
    negative_regret_traces = []
    positive_regret_traces = []
    for i, trace in enumerate(data):
        cumsum = np.cumsum(opt-np.array(trace))
        if cumsum[-1] < 0:
            negative_regret_traces.append(cumsum)
            print(np.sum(np.array(trace)>=opt))
            plt.hist(trace, bins = 25, density=True)
        else:
            positive_regret_traces.append(cumsum)
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