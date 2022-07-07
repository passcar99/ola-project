from .Learner import*
import numpy as np
from typing import List, Dict
from .GPUCB_Learner import GPUCB_Learner

class GPUCB_Learner5(GPUCB_Learner):

    def __init__(self, arms, conpam_matrix:List[Dict], prob_buy, avg_sold,margins, bounds,environment_type = 'fast'):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: data about the environment (see the environment classes).
        :param con_matrix: connectivity matrix of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param margins: margin (profit) for each arm.
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        self.con_matrix = np.ones((len(margins), len(margins)))
        self.transition_count_matrix = np.zeros((len(margins), len(margins)))
        self.prod_count = np.zeros((len(margins)))
        super().__init__(arms,conpam_matrix,self.con_matrix, prob_buy, avg_sold,margins, bounds, environment_type)

            

    def update_model(self,activation_history):
        """ 
        Update the Gaussian Processes for every product to incorporate the new data.
        Update the average number of items sold.
        Update the estimate on the graph weights.
        """
        super().update_model()
        self.prod_count += (activation_history!=0).sum(axis=0)
        for visit in range(activation_history.shape[0]):
            visit_prob = np.zeros((self.n_products, self.n_products))
            for product in range(self.n_products):
                activ_time = activation_history[visit, product]
                previous_nodes = np.logical_and(activation_history[visit] == activ_time-1, activation_history[visit]!=0)
                sum_weights = previous_nodes.sum()
                if sum_weights!=0:
                    visit_prob[previous_nodes, product] = 1/sum_weights
            self.transition_count_matrix += visit_prob
        self.con_matrix = self.transition_count_matrix/self.prod_count
        self.con_matrix[np.isnan(self.con_matrix)] = 0 # set to zero items with zero visits

        self.env.con_matrix = self.con_matrix
                
        

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model(reward['activation_history'])
        

