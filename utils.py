import matplotlib.pyplot as plt
import os
import numpy as np
import json

def plot_gaussian_process(learner,dir_name=""):
    if not(learner.t <=4 or learner.t%5==0):
        return
    os.makedirs('plots/' + dir_name, exist_ok=True)
    fig, axes = plt.subplots(learner.n_products, figsize=(9, 16))
    for product in range(learner.n_products):
        x = np.atleast_2d(learner.pulled_arms[product][-learner.sliding_window:])
        y = learner.rewards_per_product[product][-learner.sliding_window:]
        means, sigmas = learner.means[product], learner.sigmas[product]
        axes[product].plot(learner.arms, means)
        axes[product].scatter(x, y, c='b')
        axes[product].fill_between(learner.arms, means-1.96*sigmas, means+1.96*sigmas, alpha=0.7, color='c')
        axes[product].set_ylim((0, 1))
        axes[product].set_title('Product n° '+str(product))
    file_name = 'plots/'+ dir_name +'/iter'+str(learner.t)
    plt.suptitle('Iteration'+str(learner.t))
    plt.savefig(fname=file_name)
    plt.close(fig)

""" 
Function so save the rewards.
:param rewards_list: list of rewards (of any kind)
:param experiment_name: name of the experiment
:param learner_name: name of the learner
:param experiment_id: identifier of the experiment
"""
def save_rewards(rewards_list, experiment_name, learner_name, experiment_id=0):
    new_dir = 'backup/'+experiment_name+'/'
    os.makedirs(new_dir, exist_ok=True)
    filename = new_dir + learner_name + str(experiment_id)
    with open(filename, "w") as fp:
        json.dump(rewards_list, fp)


def plot_and_save_rewards(rewards_list, clairvoyant_rewards, learner_name_list, experiment_name, experiment_lenght, display_figure=True):
    fig = plt.figure(0)
    plt.ylabel("Regret, "+ experiment_name)
    plt.xlabel("t")
    clairvoyant_rewards_per_experiment = np.array(clairvoyant_rewards)
    plot_list = []
    colors = ['r', 'g', 'c', 'b', 'y']
    T = experiment_lenght
    for i, learner_rewards in enumerate(rewards_list):
        cum_reg = np.cumsum(clairvoyant_rewards_per_experiment-np.array(learner_rewards), axis = 1)
        mean = cum_reg.mean(axis=0)
        std = cum_reg.std(axis=0)
        p = plt.plot(np.arange(0, T), mean, colors[i])
        plt.fill_between(np.arange(0, T), mean-std,mean+std, alpha=0.7, color=colors[i])
        plot_list.append(p[0])
    print(plot_list)
    plt.legend(plot_list, learner_name_list)
    file_name = 'backup/'+experiment_name+'_rewards.png'
    plt.savefig(fname=file_name)
    if display_figure:
        plt.show()
    else:
        plt.close(fig)
