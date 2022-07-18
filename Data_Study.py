import json
import matplotlib.pyplot as plt
import os
from utils import *
import numpy as np


#STEP6
def Step6():
    opt=[ 5408.51196,     8849.84628,    72176.05288485] #STEP 6 FULLY
    opt=[ 5408.51196,     8849.84628,    68703.33333333]  #STEP 6 NOT FULLY, first phases equal because 0% user in TIER 1 products
    ucb_detecting_rewards_per_experiment=[]
    ucb_sliding_rewards_per_experiment=[]
    ex_ucb_detecting_rewards_per_experiment=[]
    ex_ucb_sliding_rewards_per_experiment=[]

    for i in range(3):
        path="../NON_FULLY/STEP6_"+str(i+1)+"/backup/Step6"
        with open(path+"/GPUCB_Learner_detecting_ex-1") as fp:
                local_ex_reward_detecting=json.load(fp)
        with open(path+"/GPUCB_Learner_detecting-1") as fp:
                local_reward_detecting=json.load(fp)
        ucb_detecting_rewards_per_experiment=ucb_detecting_rewards_per_experiment+local_reward_detecting
        ex_ucb_detecting_rewards_per_experiment=ex_ucb_detecting_rewards_per_experiment+local_ex_reward_detecting

        with open(path+"/GPUCB_Learner_sliding_ex-1") as fp:
                local_ex_reward_sliding=json.load(fp)
        with open(path+"/GPUCB_Learner_sliding-1") as fp:
                local_reward_sliding=json.load(fp)
        ucb_sliding_rewards_per_experiment=ucb_sliding_rewards_per_experiment+local_reward_sliding
        ex_ucb_sliding_rewards_per_experiment=ex_ucb_sliding_rewards_per_experiment+local_ex_reward_sliding


    ex_reward_Clayr=[]
    breakpoints = np.array([120,204])
    for t in range(360):
        phase = np.sum(breakpoints <= t)
        ex_reward_Clayr.append(opt[phase])

    plot_and_save_regrets([ucb_detecting_rewards_per_experiment,ucb_sliding_rewards_per_experiment],ex_reward_Clayr, [ "UCB detecting", "UCB sliding"], "TEST", 360, display_figure=True)

    learner_name_list=[ "UCB detecting", "UCB sliding","Clayrvoiant"]
    experiment_lenght=360
    clairvoyant_rewards=ex_reward_Clayr
    rewards_list=[ex_ucb_detecting_rewards_per_experiment,ex_ucb_sliding_rewards_per_experiment]

    fig = plt.figure(0)
    plt.ylabel("Expected Reward")
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.array(learner_rewards)
        mean = cum_reg.mean(axis=0)
        std = cum_reg.std(axis=0)
        z_sqrt_n = 1.96 / np.sqrt(len(cum_reg))
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-z_sqrt_n*std,mean+z_sqrt_n *std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    CC=plt.plot(np.arange(0, T),ex_reward_Clayr,colors[-2])
    plot_list.append(CC)
    plt.legend(plot_list, learner_name_list)
    plt.show()

#STEP3
def Step3():
    opt= 72176.05288485333 #STEP 3 4 5 FULL
    opt = 68703.33333333 #NO FULL
    T=160 

    ts_rewards_per_experiment=[]
    ucb_rewards_per_experiment=[]
    path="../STEP3/backup/Step3"
    with open(path+"/GPTS_Learner-1") as fp:
                local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner-1") as fp:
                local_ucb_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward

    path="../step3-2/backup/Step3"
    with open(path+"/GPTS_Learner-1") as fp:
                local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner-1") as fp:
                local_ucb_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward


    clairvoyant_rewards_per_experiment=[opt]*T

    plot_and_save_regrets([ts_rewards_per_experiment,ucb_rewards_per_experiment],clairvoyant_rewards_per_experiment, ["TS", "UCB"], "TEST", T, display_figure=True)

    learner_name_list=[ "TS", "UCB","Clayrvoiant"]
    experiment_lenght=160
    clairvoyant_rewards=clairvoyant_rewards_per_experiment
    rewards_list=[ts_rewards_per_experiment,ucb_rewards_per_experiment]

    fig = plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.array(learner_rewards)
        mean = cum_reg.mean(axis=0)
        std = cum_reg.std(axis=0)
        z_sqrt_n = 1.96 / np.sqrt(len(cum_reg))
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-z_sqrt_n*std,mean+z_sqrt_n *std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    CC=plt.plot(np.arange(0, T),clairvoyant_rewards_per_experiment,colors[-2])
    plot_list.append(CC)
    plt.legend(plot_list, learner_name_list)
    plt.show()

#STEP4
def Step4():
    opt= 72176.05288485333 #STEP 3 4 5
    opt = 68703.33333333 #NO FULL
    T=160

    ts_rewards_per_experiment=[]
    ucb_rewards_per_experiment=[]
    path="../STEP4/backup/Step4"
    with open(path+"/GPTS_Learner4-1") as fp:
                local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner4-1") as fp:
                local_ucb_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward

    path="../step4-1/backup/Step4"
    with open(path+"/GPTS_Learner4-1") as fp:
                local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner4-1") as fp:
                local_ucb_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward


    clairvoyant_rewards_per_experiment=[opt]*T

    plot_and_save_regrets([ts_rewards_per_experiment,ucb_rewards_per_experiment],clairvoyant_rewards_per_experiment, ["TS", "UCB"], "TEST", T, display_figure=True)


    learner_name_list=[ "TS", "UCB","Clayrvoiant"]
    experiment_lenght=160
    clairvoyant_rewards=clairvoyant_rewards_per_experiment
    rewards_list=[ts_rewards_per_experiment,ucb_rewards_per_experiment]

    fig = plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.array(learner_rewards)
        mean = cum_reg.mean(axis=0)
        std = cum_reg.std(axis=0)
        z_sqrt_n = 1.96 / np.sqrt(len(cum_reg))
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-z_sqrt_n*std,mean+z_sqrt_n *std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    CC=plt.plot(np.arange(0, T),clairvoyant_rewards_per_experiment,colors[-2])
    plot_list.append(CC)
    plt.legend(plot_list, learner_name_list)
    plt.show()


#STEP5
def Step5():
    opt= 72176.05288485333 #STEP 3 4 5
    #opt = 68703.33333333 #NO FULL
    T=160

    ts_rewards_per_experiment=[]
    ucb_rewards_per_experiment=[]
    path="../STEP5/backup/Step5"
    #with open(path+"/GPTS_Learner5-1") as fp:
    #            local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner5-1") as fp:
                local_ucb_reward=json.load(fp)
    #ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward

    path="../step5-1/backup/Step5"
    with open(path+"/GPTS_Learner5-1") as fp:
                local_ts_reward=json.load(fp)
    with open(path+"/GPUCB_Learner5-1") as fp:
                local_ucb_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward
    ucb_rewards_per_experiment=ucb_rewards_per_experiment+local_ucb_reward


    clairvoyant_rewards_per_experiment=[opt]*T

    plot_and_save_regrets([ts_rewards_per_experiment,ucb_rewards_per_experiment],clairvoyant_rewards_per_experiment, ["TS", "UCB"], "TEST", T, display_figure=True)

    learner_name_list=[ "TS", "UCB","Clayrvoiant"]
    experiment_lenght=160
    clairvoyant_rewards=clairvoyant_rewards_per_experiment
    rewards_list=[ts_rewards_per_experiment,ucb_rewards_per_experiment]

    fig = plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.array(learner_rewards)
        mean = cum_reg.mean(axis=0)
        std = cum_reg.std(axis=0)
        z_sqrt_n = 1.96 / np.sqrt(len(cum_reg))
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-z_sqrt_n*std,mean+z_sqrt_n *std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    CC=plt.plot(np.arange(0, T),clairvoyant_rewards_per_experiment,colors[-2])
    plot_list.append(CC)
    plt.legend(plot_list, learner_name_list)
    plt.show()


def General():
    opt= 37607.56314192

    T=100

    ts_rewards_per_experiment=[]
    ucb_rewards_per_experiment=[]
    path="../ContextManagerTS-1"

    with open(path) as fp:
                local_ts_reward=json.load(fp)
    ts_rewards_per_experiment=ts_rewards_per_experiment+local_ts_reward

    clairvoyant_rewards_per_experiment=[opt]*T

    plot_and_save_regrets([ts_rewards_per_experiment],clairvoyant_rewards_per_experiment, ["TS"], "TEST", T, display_figure=True)

    learner_name_list=[ "TS","Clayrvoiant"]
    experiment_lenght=100
    clairvoyant_rewards=clairvoyant_rewards_per_experiment
    rewards_list=[ts_rewards_per_experiment]

    fig = plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.array(learner_rewards)
        mean = cum_reg.mean(axis=0)
        print(mean.shape)
        std = cum_reg.std(axis=0)
        z_sqrt_n = 1.96 / np.sqrt(len(cum_reg))
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-z_sqrt_n*std,mean+z_sqrt_n *std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    CC=plt.plot(np.arange(0, T),clairvoyant_rewards_per_experiment,colors[-2])
    plot_list.append(CC)
    plt.legend(plot_list, learner_name_list)
    plt.show()

General()