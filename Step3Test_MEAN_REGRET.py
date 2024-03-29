from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner import GPTS_Learner
from learners.TSLearnerTopped5D import GPTS_Learner_TOP5D
from learners.GPUCB_Learner import GPUCB_Learner
from environment.Environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from environment.Algorithms import budget_allocations
import sys
from utils import plot_gaussian_process



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
    conpam_matrix = [
        {"alpha_params": [(0, 30, 50), (0, 25, 5),(5, 20, 10),(5, 40, 15),(5, 25, 20)], 
        "features":[0, 0], "total_mass":100, "avg_number":100}, 
                    ]
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
    Clayrvoiant=np.zeros(pow(n_arms,5))
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
    opt = max(Clayrvoiant)
    #print(expected_margin)

    ts_rewards_per_experiment = []
    ucb_rewards_per_experiment = []
    tsTOP5D_rewards_per_experiment = []
    n_experiments = 1

    T = 100

    Regret_ts=np.zeros(T+1)
    Regret_5D=np.zeros(T+1)
    Regret_ucb=np.zeros(T+1)

    Ex_Reward_ts=np.zeros(T)
    Ex_Reward_5D=np.zeros(T)
    Ex_Reward_ucb=np.zeros(T)

    for e in tqdm(range(n_experiments)):
        env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ts_learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        tsTOP5D_learner = GPTS_Learner_TOP5D(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        ucb_learner = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        ts_learner.avg_n_users = 100
        tsTOP5D_learner.avg_n_users = 100
        ucb_learner.avg_n_users = 100
        for t in tqdm(range(0, T)):
            pulled_arm_ts = ts_learner.pull_arm()
            #pulled_arm_5D = tsTOP5D_learner.pull_arm()
            pulled_arm_ucb = ucb_learner.pull_arm()

            pulled_arm_ts_list=list(pulled_arm_ts)
            #pulled_arm_5D_list=list(pulled_arm_5D)
            pulled_arm_ucb_list=list(pulled_arm_ucb)

            arms_list=list(arms)

            pulled_arm_ts_idx=[arms_list.index(pulled_arm_ts_list[0]),arms_list.index(pulled_arm_ts_list[1]),arms_list.index(pulled_arm_ts_list[2]),arms_list.index(pulled_arm_ts_list[3]),arms_list.index(pulled_arm_ts_list[4])]
            #pulled_arm_5D_idx=[arms_list.index(pulled_arm_5D_list[0]),arms_list.index(pulled_arm_5D_list[1]),arms_list.index(pulled_arm_5D_list[2]),arms_list.index(pulled_arm_5D_list[3]),arms_list.index(pulled_arm_5D_list[4])]
            pulled_arm_ucb_idx=[arms_list.index(pulled_arm_ucb_list[0]),arms_list.index(pulled_arm_ucb_list[1]),arms_list.index(pulled_arm_ucb_list[2]),arms_list.index(pulled_arm_ucb_list[3]),arms_list.index(pulled_arm_ucb_list[4])]
            
            Number_ts=pulled_arm_ts_idx[0]+n_arms*pulled_arm_ts_idx[1]+pow(n_arms,2)*pulled_arm_ts_idx[2]+pow(n_arms,3)*pulled_arm_ts_idx[3]+pow(n_arms,4)*pulled_arm_ts_idx[4]
            #Number_5D=pulled_arm_5D_idx[0]+n_arms*pulled_arm_5D_idx[1]+pow(n_arms,2)*pulled_arm_5D_idx[2]+pow(n_arms,3)*pulled_arm_5D_idx[3]+pow(n_arms,4)*pulled_arm_5D_idx[4]
            Number_ucb=pulled_arm_ucb_idx[0]+n_arms*pulled_arm_ucb_idx[1]+pow(n_arms,2)*pulled_arm_ucb_idx[2]+pow(n_arms,3)*pulled_arm_ucb_idx[3]+pow(n_arms,4)*pulled_arm_ucb_idx[4]

            Regret_ts[t+1]=Regret_ts[t]+max(Clayrvoiant)-Clayrvoiant[Number_ts]
            #Regret_5D[t+1]=Regret_5D[t]+max(Clayrvoiant)-Clayrvoiant[Number_5D]
            Regret_ucb[t+1]=Regret_ucb[t]+max(Clayrvoiant)-Clayrvoiant[Number_ucb]

            Ex_Reward_ts[t]=Clayrvoiant[Number_ts]
            #Ex_Reward_5D[t]=Clayrvoiant[Number_5D]
            Ex_Reward_ucb[t]=Clayrvoiant[Number_ucb]

            reward_ts = env.round(pulled_arm_ts)
            #reward_5D = env.round(pulled_arm_5D)
            reward_ucb = env.round(pulled_arm_ucb)

            ts_learner.update(pulled_arm_ts, reward_ts[0])
            #tsTOP5D_learner.update(pulled_arm_5D, reward_5D[0])
            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            plot_gaussian_process(ts_learner)


        ts_rewards_per_experiment.append(ts_learner.collected_rewards)
        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        #tsTOP5D_rewards_per_experiment.append(tsTOP5D_learner.collected_rewards)

    idx_max=np.argmax(Clayrvoiant)
    clayt_opt=[idx_max%n_arms,idx_max%pow(n_arms,2)//n_arms,idx_max%pow(n_arms,3)//pow(n_arms,2),idx_max%pow(n_arms,4)//pow(n_arms,3),idx_max//pow(n_arms,4)]
    print(clayt_opt)
    print(opt)

    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.arange(0, T+1), Regret_ts, 'r')
    #plt.plot(np.arange(0, T+1), Regret_5D, 'b')
    plt.plot(np.arange(0, T+1), Regret_ucb, 'g')

    plt.legend(["TS", "UCB"])
    plt.show()

    Ex_Reward_Clayr=np.zeros(T)
    Ex_Reward_Clayr=Ex_Reward_Clayr+opt

    plt.figure(1)
    plt.ylabel("Expected reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), Ex_Reward_ts, 'r')
    #plt.plot(np.arange(0, T), Ex_Reward_5D, 'b')
    plt.plot(np.arange(0, T), Ex_Reward_ucb, 'g')
    plt.plot(np.arange(0, T), Ex_Reward_Clayr, 'c')
    plt.legend(["TS", "UCB","Clayr"])
    plt.show()

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """