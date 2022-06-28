import numpy as np
from typing import List, Dict

 # TODO modify later to take into account the subclasses
class Learner():

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds):
        self.arms: List = arms
        self.n_products = len(con_matrix)
        self.n_arms = len(arms)
        self.t = 0
        self.rewards_per_arm = [[[] for a in range(self.n_arms)] for p in range(self.n_products)]
        self.collected_rewards = []
        self.conpam_matrix = conpam_matrix
        self.con_matrix = con_matrix
        self.prob_buy = prob_buy
        self.avg_sold = avg_sold
        self.margins = margins
        self.bounds = bounds
        self.unfeasible_arms = []
        for p in range(self.n_products):
            self.unfeasible_arms.append(np.logical_or(self.arms <= self.bounds[p][0], self.arms >= self.bounds[p][1]))



    def update_observations(self, pulled_arms, reward):
        alphas = reward['alphas']
        for product in range(self.n_products):
            arm = np.where(self.arms==pulled_arms[product])[0][0]
            self.rewards_per_arm[product][arm].append(alphas[product+1]) # ignore alpha0
        self.collected_rewards.append(np.sum(reward['profit']))

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.update_model()