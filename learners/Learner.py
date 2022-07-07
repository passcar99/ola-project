import numpy as np
from typing import List, Dict

 # TODO modify later to take into account the subclasses
class Learner():

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: data about the environment (see the environment classes).
        :param con_matrix: connectivity matrix of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param avg_sold: average quantity of items bought for each product when displayed as primary.
        :param margins: margin (profit) for each arm.
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        self.arms = arms
        self.n_products = len(margins)
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
        self.avg_n_users = 0



    def update_observations(self, pulled_arms, reward):
        alphas = reward['alphas']
        for product in range(self.n_products):
            arm = np.where(self.arms==pulled_arms[product])[0][0]
            self.rewards_per_arm[product][arm].append(alphas[product+1]) # ignore alpha0
        self.collected_rewards.append(np.sum(reward['profit']))
        self.avg_n_users = (self.avg_n_users*(self.t-1)+reward['n_users'])/self.t

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.update_model()