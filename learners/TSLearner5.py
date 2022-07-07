import numpy as np
from typing import List, Dict
from environment.RandomEnvironment import RandomEnvironment
from .TSLearner import GPTS_Learner



class GPTS_Learner5(GPTS_Learner):
    """ 
    Gaussian Process Thompson Sampling bandit. It estimated the click rate(?) of each product independently.
    At every round it computes the alphas and the expected margins and fills the DP table. Then selects one of the 
    feasible superarm. The average number of items sold is estimated according to data
    
    """
    def __init__(self, arms, conpam_matrix:List[Dict], prob_buy,avg_sold, margins, bounds,environment_type = 'fast'):
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
        self.prod_count += (activation_history>0).sum(axis=0)
        for visit in range(activation_history.shape[0]):
            visit_prob = np.zeros((self.n_products, self.n_products))
            for product in range(self.n_products):
                activ_time = activation_history[visit, product]
                previous_nodes = np.logical_and(activation_history[visit] == activ_time-1,np.logical_and(activation_history[visit]!=0, items_sold[visit]!=0))
                sum_weights = previous_nodes.sum()
                if sum_weights != 0:
                    visit_prob[previous_nodes, product] = (1/sum_weights)#/self.prob_buy[previous_nodes]
            self.credit_matrix += visit_prob
        non_zero_prods = (self.prod_count!=0)
        self.con_matrix[:, non_zero_prods] = self.credit_matrix[:, non_zero_prods]/self.prod_count[non_zero_prods]
        #self.con_matrix[np.isnan(self.con_matrix)] = 0 # put to zero items with zero visits
        self.env.con_matrix = self.con_matrix
                
        

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model(reward['items'],reward['activation_history'])


        



if __name__ == '__main__':
    connectivity_matrix = np.array([[0, 0.2, 0.4, 0.3, 0.1],
                                    [0.5, 0, 0.1, 0.3, 0.1],
                                    [0.3, 0.2, 0, 0.1, 0.4],
                                    [0.13, 0.17, 0.30, 0, 0.4],
                                    [0.16, 0.34, 0.15, 0.25, 0],
                                    ])
    prob_buy = np.array([0.1, 0.2, 0.5, 0.9, 0.7])
    avg_sold = [5,6,7,8,9]
    margins = [10, 20, 30, 40, 50]
    conpam_matrix = [{"alpha_params": [(0, 10, 2), (5, 10, 6),(5, 20, 10),(5, 50, 6),(5, 8, 6)], "features":[0, 0], "total_mass":64, "avg_number":100}, 
                    ]
    env = RandomEnvironment(conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    arms = np.array([0, 5, 10, 15, 20, 25])
    bounds = np.array([[-1, 100],[-1, 100],[-1, 100],[-1, 100],[-1, 100]])
    learner = GPTS_Learner5(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

    for _ in range(100):
        arm = learner.pull_arm()
        print(arm)
        feedback = env.round(arm.flatten())

        learner.update(arm, feedback[0])
        #TODO

