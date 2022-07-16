import numpy as np
from typing import List, Dict
from environment.RandomEnvironment import RandomEnvironment
from .TSLearner import GPTS_Learner



class GPTS_Learner4(GPTS_Learner):

    NAME = "GPTS_Learner4"
    
    """ 
    Gaussian Process Thompson Sampling bandit. It estimated the click rate(?) of each product independently.
    At every round it computes the alphas and the expected margins and fills the DP table. Then selects one of the 
    feasible superarm. The average number of items sold is estimated according to data
    
    """
    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, margins, bounds,environment_type = 'fast'):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: data about the environment (see the environment classes).
        :param con_matrix: connectivity matrix of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param margins: margin (profit) for each arm.
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        self.avg_sold = np.ones((len(con_matrix))) # dummy number for just the first iteration
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, self.avg_sold, margins, bounds, environment_type)
        self.tot_visits = 0 # total number of visits so far



    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model(reward['items'])

    def update_model(self, items_sold):
        """ 
        Update the Gaussian Processes for every product to incorporate the new data.
        Update the average number of items sold.
        """
        super().update_model()
        today_visits = items_sold.shape[0]

        today_avg_sold = np.nanmean(items_sold, axis=0)
        mask = np.logical_not(np.isnan(today_avg_sold))
        #running average to be more efficient. Only at the first iteration tot_visits==0, consequently avg_sold is ignored
        self.avg_sold[mask] = (self.avg_sold[mask] * self.tot_visits + today_avg_sold[mask]* today_visits)/(self.tot_visits + today_visits)
        self.tot_visits += today_visits
        self.env.avg_sold = self.avg_sold


        



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
    learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

    for _ in range(100):
        arm = learner.pull_arm()
        print(arm)
        feedback = env.round(arm.flatten())

        learner.update(arm, feedback[0])
        #TODO

