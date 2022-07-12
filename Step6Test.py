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
import math
from utils import plot_gaussian_process, save_rewards, plot_and_save_rewards
import warnings
warnings.filterwarnings("ignore")


EXPERIMENT_NAME = "Step6"
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

    breakpoints = np.array([25])
    conpam_matrix = [
        {"alpha_params": [
                            [(0, 5, 20), (0, 5, 5),(5, 15, 10),(5, 15, 15),(5, 15, 20)],
                            #[(0, 10, 20), (2, 15, 20),(2, 20, 20),(2, 15, 20),(1, 15, 20)],
                            [(5, 15, 20), (5, 15, 20),(0, 5, 20),(0, 5, 20),(0, 5, 20)]
                            #[(0, 2, 20), (3, 10, 20),(0, 5, 20),(5, 17, 20),(2, 4, 20)]
                         ], 
        "features":[0, 0], "total_mass":100, "avg_number":100, "breakpoints":breakpoints} 
                    ]#Certainly will vary
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
    Clayrvoiant=np.zeros([pow(n_arms,5),len(breakpoints)+1])#rows superarm, columnns phases
    for phase_num in range(len(breakpoints)+1):
        ii=0
        for m in range(n_arms):
            for l in range(n_arms):
                for k in range(n_arms):
                    for j in range(n_arms):
                        for i in range(n_arms):
                            sys.stdout.write('\r')
                            super_arm=[arms[i],arms[j],arms[k],arms[l],arms[m]]
                            arms_list=list(arms)
                            super_arm_idx=[arms_list.index(arms[i]),arms_list.index(arms[j]),arms_list.index(arms[k]),arms_list.index(arms[l]),arms_list.index(arms[m])]
                            if super_arm_idx in unfeasible_arms:
                                Clayrvoiant[ii,phase_num]=-np.inf
                            else:
                                time= breakpoints[phase_num-1]+1 if phase_num==len(breakpoints) else breakpoints[phase_num]-1
                                Clayrvoiant[ii,phase_num]=envEx.pull_arm_excpected_value(super_arm,time)-sum(super_arm)
                            sys.stdout.write("Clayrvoiant: {:.1f}%".format((100/(pow(n_arms,5)-1)*ii)))
                            sys.stdout.flush()
                            ii+=1
    opt = Clayrvoiant.max(0)#opt row vector with opt for every phase
    idx_max_phase1=np.argmax(Clayrvoiant[:,0])
    idx_max_phase2=np.argmax(Clayrvoiant[:,1])
    clayt_opt1=[idx_max_phase1%n_arms,idx_max_phase1%pow(n_arms,2)//n_arms,idx_max_phase1%pow(n_arms,3)//pow(n_arms,2),idx_max_phase1%pow(n_arms,4)//pow(n_arms,3),idx_max_phase1//pow(n_arms,4)]
    clayt_opt2=[idx_max_phase2%n_arms,idx_max_phase2%pow(n_arms,2)//n_arms,idx_max_phase2%pow(n_arms,3)//pow(n_arms,2),idx_max_phase2%pow(n_arms,4)//pow(n_arms,3),idx_max_phase2//pow(n_arms,4)]
    print("Optimum phase 1: ")
    print(clayt_opt1)
    print("Optimum phase 2: ")
    print(clayt_opt2)


    ucb_rewards_per_experiment = []
    ucb_detecting_rewards_per_experiment = []
    ucb_sliding_rewards_per_experiment = []
    n_experiments = 1

    T = 70

    regret_ucb=np.zeros(T+1)
    regret_ucb_detecting=np.zeros(T+1)
    regret_ucb_sliding=np.zeros(T+1)

    ex_reward_ucb=np.zeros(T)
    ex_reward_ucb_detecting=np.zeros(T)
    ex_reward_ucb_sliding=np.zeros(T)

    for e in tqdm(range(n_experiments)):
        env_ucb = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ucb_learner = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')
        env_ucb_detecting = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ucb_learner_detecting = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast','detect')
        env_ucb_sliding = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
        ucb_learner_sliding = GPUCB_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast','slide',int(2*math.sqrt(T)))
        ucb_learner.avg_n_users = 100
        ucb_learner_detecting.avg_n_users = 100
        ucb_learner_sliding.avg_n_users = 100
        for t in tqdm(range(0, T)):
            pulled_arm_ucb = ucb_learner.pull_arm()
            pulled_arm_detecting = ucb_learner_detecting.pull_arm()
            pulled_arm_ucb_sliding = ucb_learner_sliding.pull_arm()

            pulled_arm_ucb=list(pulled_arm_ucb)
            pulled_arm_detecting=list(pulled_arm_detecting)
            pulled_arm_ucb_sliding=list(pulled_arm_ucb_sliding)

            arms_list=list(arms)

            pulled_arm_ucb_idx=[arms_list.index(pulled_arm_ucb[0]),arms_list.index(pulled_arm_ucb[1]),arms_list.index(pulled_arm_ucb[2]),arms_list.index(pulled_arm_ucb[3]),arms_list.index(pulled_arm_ucb[4])]
            pulled_arm_ucb_det_idx=[arms_list.index(pulled_arm_detecting[0]),arms_list.index(pulled_arm_detecting[1]),arms_list.index(pulled_arm_detecting[2]),arms_list.index(pulled_arm_detecting[3]),arms_list.index(pulled_arm_detecting[4])]
            pulled_arm_ucb_sliding_idx=[arms_list.index(pulled_arm_ucb_sliding[0]),arms_list.index(pulled_arm_ucb_sliding[1]),arms_list.index(pulled_arm_ucb_sliding[2]),arms_list.index(pulled_arm_ucb_sliding[3]),arms_list.index(pulled_arm_ucb_sliding[4])]
            
            number_ucb=pulled_arm_ucb_idx[0]+n_arms*pulled_arm_ucb_idx[1]+pow(n_arms,2)*pulled_arm_ucb_idx[2]+pow(n_arms,3)*pulled_arm_ucb_idx[3]+pow(n_arms,4)*pulled_arm_ucb_idx[4]
            number_detecting=pulled_arm_ucb_det_idx[0]+n_arms*pulled_arm_ucb_det_idx[1]+pow(n_arms,2)*pulled_arm_ucb_det_idx[2]+pow(n_arms,3)*pulled_arm_ucb_det_idx[3]+pow(n_arms,4)*pulled_arm_ucb_det_idx[4]
            number_ucb_sliding=pulled_arm_ucb_sliding_idx[0]+n_arms*pulled_arm_ucb_sliding_idx[1]+pow(n_arms,2)*pulled_arm_ucb_sliding_idx[2]+pow(n_arms,3)*pulled_arm_ucb_sliding_idx[3]+pow(n_arms,4)*pulled_arm_ucb_sliding_idx[4]

            phase = np.sum(breakpoints <= t)

            regret_ucb[t+1]=regret_ucb[t]+opt[phase]-Clayrvoiant[number_ucb][phase]
            regret_ucb_detecting[t+1]=regret_ucb_detecting[t]+opt[phase]-Clayrvoiant[number_detecting][phase]
            regret_ucb_sliding[t+1]=regret_ucb_sliding[t]+opt[phase]-Clayrvoiant[number_ucb_sliding][phase]

            ex_reward_ucb[t]=Clayrvoiant[number_ucb][phase]
            ex_reward_ucb_detecting[t]=Clayrvoiant[number_detecting][phase]
            ex_reward_ucb_sliding[t]=Clayrvoiant[number_ucb_sliding][phase]

            reward_ucb = env_ucb.round(pulled_arm_ucb)
            reward_ucb_detecting = env_ucb_detecting.round(pulled_arm_detecting)
            reward_ucb_sliding = env_ucb_sliding.round(pulled_arm_ucb_sliding)

            ucb_learner.update(pulled_arm_ucb, reward_ucb[0])
            ucb_learner_detecting.update(pulled_arm_detecting, reward_ucb_detecting[0])
            ucb_learner_sliding.update(pulled_arm_ucb_sliding, reward_ucb_sliding[0])

            plot_gaussian_process(ucb_learner_detecting)

        ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
        ucb_detecting_rewards_per_experiment.append(ucb_learner_sliding.collected_rewards)
        ucb_sliding_rewards_per_experiment.append(ucb_learner_detecting.collected_rewards)

    save_rewards(ucb_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner.NAME, -1)
    save_rewards(ucb_detecting_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner_detecting.NAME+'_detecting', -1)
    save_rewards(ucb_sliding_rewards_per_experiment, EXPERIMENT_NAME, ucb_learner_sliding.NAME+'_sliding', -1)

    plt.figure(0)
    plt.ylabel("regret")
    plt.xlabel("t")
    plt.plot(np.arange(0, T+1), regret_ucb, 'r')
    plt.plot(np.arange(0, T+1), regret_ucb_detecting, 'b')
    plt.plot(np.arange(0, T+1), regret_ucb_sliding, 'g')
    plt.legend(["UCB", "UCB_detecting", "UCB_sliding"])
    plt.show()

    ex_reward_Clayr=np.zeros(T)
    ex_reward_Clayr[0:breakpoints[0]]=opt[0]
    for brk_idx in range(len(breakpoints)-1):
        ex_reward_Clayr[breakpoints[i]:breakpoints[i+1]]=opt[i]
    ex_reward_Clayr[breakpoints[-1]:]=opt[-1]

    fig = plt.figure(1)
    plt.ylabel("Expected reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), ex_reward_ucb, 'r')
    plt.plot(np.arange(0, T), ex_reward_ucb_detecting, 'b')
    plt.plot(np.arange(0, T), ex_reward_ucb_sliding, 'g')
    plt.plot(np.arange(0, T), ex_reward_Clayr, 'c')
    plt.legend(["UCB vanilla", "UCB detecting", "UCB sliding","Clayr"])

    plot_and_save_rewards([ucb_rewards_per_experiment,ucb_detecting_rewards_per_experiment,ucb_sliding_rewards_per_experiment],
                        ex_reward_Clayr, ["UCB vanilla", "UCB detecting", "UCB sliding",], EXPERIMENT_NAME, T, display_figure=DISPLAY_FIGURE)

    """ plt.figure(1)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.arange(0, T), np.mean(clairvoyant_rewards_per_experiment, axis = 0), 'b')
    plt.plot(np.arange(0, T), np.mean(ts_rewards_per_experiment, axis = 0), 'r')
    plt.legend(["Clairvoyant", "TS"])
    plt.show() """