import numpy as np
from typing import List, Dict

 # TODO modify later to take into account the subclasses
class Learner():

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds):
        self.arms: List = arms
        self.n_products = len(con_matrix)
        self.n_arms = len(arms)
        self.t = 0
        self.rewards_per_arm = [[[] for a in range(self.arms)] for p in range(self.n_products)]
        self.collected_rewards = np.array([])
        self.conpam_matrix = conpam_matrix
        self.con_matrix = con_matrix
        self.prob_buy = prob_buy
        self.avg_sold = avg_sold
        self.margins = margins
        self.bounds = bounds


    def update_observations(self, pulled_arms, reward):
        for product in range(self.n_products):
            arm = self.arms.index(pulled_arms[product])
            self.rewards_per_arm[product][arm].append(reward[product+1]) # ignore alpha0
        self.collected_rewards = np.append(self.collected_rewards, reward)

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.update_model()