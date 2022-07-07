from .Learner import*
import numpy as np
from typing import List, Dict
from .GPUCB_Learner import GPUCB_Learner

class GPUCB_Learner5(GPUCB_Learner):

    def __init__(self, arms, conpam_matrix:List[Dict], conn_constr_matrix, prob_buy,avg_sold, margins, bounds,environment_type = 'fast'):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: data about the environment (see the environment classes).
        :param conn_constr_matrix: connectivity mask of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param margins: margin (profit) for each arm.
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        self.constraints_matrix = conn_constr_matrix
        self.con_matrix = self.constraints_matrix * 1.0
        self.credit_matrix = np.zeros((len(margins), len(margins)))
        self.prod_count = np.zeros((len(margins), len(margins)))
        super().__init__(arms,conpam_matrix,self.con_matrix, prob_buy, avg_sold, margins, bounds, environment_type)

            

    def update_model(self, items_sold, activation_history):
        """ 
        Update the Gaussian Processes for every product to incorporate the new data.
        Update the average number of items sold.
        Update the estimate on the graph weights.
        """
        super().update_model()

        for visit in range(activation_history.shape[0]):
            visit_prob = np.zeros((self.n_products, self.n_products))
            active_nodes = np.logical_and(activation_history[visit]!=0, items_sold[visit]!=0) > 0
            for product in range(self.n_products):
                activ_time = activation_history[visit, product]
                previous_nodes = np.logical_and(activation_history[visit] == activ_time-1,active_nodes)
                sum_weights = previous_nodes.sum()
                if sum_weights != 0:
                    visit_prob[previous_nodes, product] = (1/sum_weights)#/self.prob_buy[previous_nodes]
                other_prods = np.arange(self.n_products)!=product
                self.prod_count[other_prods, product] += (previous_nodes)[other_prods]>0 # active before current node
                self.prod_count[other_prods, product] += (np.logical_and(active_nodes, activ_time==0))[other_prods] # active but current node isn't
            self.credit_matrix += visit_prob
        non_zero_prods = np.logical_and((self.prod_count!=0), self.constraints_matrix!=0)
        self.con_matrix[non_zero_prods] = self.credit_matrix[non_zero_prods]/self.prod_count[non_zero_prods]
        self.env.con_matrix = self.con_matrix
                
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model(reward['items'],reward['activation_history'])


