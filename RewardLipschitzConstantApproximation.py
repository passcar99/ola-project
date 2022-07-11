from configparser import Interpolation
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
    avg_sold = [2,4,1.5,2,3]
    margins = [1000, 300, 100, 75, 30]

    arms = np.array([0, 5, 10, 15, 20, 25, 30])
    #bounds = np.array([[5, 100],[0, 80],[0, 50],[20, 100],[0, 100]])
    bounds = np.array([[2, 100],[2, 100],[-1, 100],[2, 100],[-1, 100]])

    env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    
    n_products = len(connectivity_matrix)
    n_arms = len(arms)
    unfeasible_arms = []
    for m in range(len(arms)):
        for l in range(len(arms)):
            for k in range(len(arms)):
                for j in range(len(arms)):
                    for i in range(len(arms)):
                        if (arms[i]+arms[j]+arms[k]+arms[l]+arms[m]>arms[-1] 
                        or arms[i]<bounds[0][0] 
                        or arms[i]>bounds[0][1]
                        or arms[j]<bounds[1][0] 
                        or arms[j]>bounds[1][1]
                        or arms[k]<bounds[2][0] 
                        or arms[k]>bounds[2][1]
                        or arms[l]<bounds[3][0] 
                        or arms[l]>bounds[3][1]
                        or arms[m]<bounds[4][0] 
                        or arms[m]>bounds[4][1]
                        ):
                            unfeasible_arms.append([i,j,k,l,m])

    #All Expected Values of the superarm
    envEx=Environment(conpam_matrix,connectivity_matrix,prob_buy,avg_sold,margins)
    Interpolation=np.zeros((n_arms,n_arms,n_arms,n_arms,n_arms))
    ii=0
    for m in range(n_arms):
        for l in range(n_arms):
            for k in range(n_arms):
                for j in range(n_arms):
                    for i in range(n_arms):
                        sys.stdout.write('\r')
                        super_arm=[arms[i],arms[j],arms[k],arms[l],arms[m]]
                        arms_list=list(arms)
                        super_arm_idx=[arms_list.index(arms[i]),arms_list.index(arms[j]),arms_list.index(arms[k]),arms_list.index(arms[l]),arms_list.index(arms[m]),]
                        if super_arm_idx in unfeasible_arms:
                            Clayrvoiant[ii]=-np.inf
                        else:
                            Clayrvoiant[ii]=envEx.pull_arm_excpected_value(super_arm,1)-sum(super_arm)
                        sys.stdout.write("Clayrvoiant: {:.1f}%".format((100/(pow(n_arms,5)-1)*ii)))
                        sys.stdout.flush()
                        ii+=1