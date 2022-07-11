import matplotlib.pyplot as plt
import os
import numpy as np

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
