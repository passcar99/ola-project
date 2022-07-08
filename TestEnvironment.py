from environment.Environment import Environment
import matplotlib.pyplot as plt
import numpy as np
import sys


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
        {"alpha_params": [(0, 10, 20), (2, 15, 20),(2, 20, 20),(2, 15, 20),(1, 15, 20)], 
        "features":[0, 0], "total_mass":100, "avg_number":100}, 
                    ]

    env=Environment(conpam_matrix,connectivity_matrix,prob_buy,avg_sold,margins)
    res0=env.site_landing(0,np.ones((5,1)))
    res1=env.site_landing(1,np.ones((5,1)))
    res2=env.site_landing(2,np.ones((5,1)))
    res3=env.site_landing(3,np.ones((5,1)))
    res4=env.site_landing(4,np.ones((5,1)))
    print(res0)
    print(res1)
    print(res2)
    print(res3)
    print(res4)

    alpha=[0.5, 0.1, 0.1, 0.15, 0.05, 0.1]
    ProbDistribution=alpha[1]*res0+alpha[2]*res1+alpha[3]*res2+alpha[4]*res3+alpha[5]*res4
    print(ProbDistribution)
    print(sum(ProbDistribution))
