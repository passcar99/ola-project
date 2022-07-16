import json
import matplotlib.pyplot as plt
import numpy as np
from environment.Algorithms import regret_bound




if __name__ == "__main__":
    filename = "backup/BackupStep5/backup/Step5-2022-07-15 19:07:39.465395/GPTS_Learner5-1"
    data =0
    with open(filename) as fp:
        data = json.load(fp)
    opt = 81280.42671666665
    negative_regret_traces = []
    for i, trace in enumerate(data):
        cumsum = np.cumsum(opt-np.array(trace))
        if cumsum[-1] < 0:
            negative_regret_traces.append(cumsum)
            print(np.sum(np.array(trace)>=opt))
            plt.hist(trace, bins = 25, density=True)
    plt.axvline(opt)
    #plt.plot(np.arange(0, 100), [opt]*100)
    
    plt.show()
    for trace in negative_regret_traces:
        plt.plot(np.array(trace))
    plt.show()

    mean = np.mean(data,axis=0)
    variance = np.var(data)
    regret = regret_bound(100*5*5, variance, 100)
    t = np.arange(1, 101)
    plt.plot(mean/regret)
    plt.show()