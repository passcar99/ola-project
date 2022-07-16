from .Learner import*
import numpy as np
from typing import List, Dict
from .GPUCB_Learner import GPUCB_Learner

class GPUCB_Learner4(GPUCB_Learner):

    NAME = "GPUCB_Learner4"
    
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
            

    def update_model(self, items_sold):
        """ 
        Update the Gaussian Processes for every product to incorporate the new data.
        Update the average number of items sold.
        """
        super().update_model()
        today_visits = np.sum(np.logical_not(np.isnan(items_sold)), axis=0)
        today_sum_sold = np.nansum(items_sold, axis=0)
        #running average to be more efficient. Only at the first iteration tot_visits==0, consequently avg_sold is ignored
        self.avg_sold[today_visits!=0] = ((self.avg_sold * self.tot_visits + today_sum_sold)/(self.tot_visits + today_visits))[today_visits!=0]
        self.tot_visits += today_visits
        self.env.avg_sold = self.avg_sold
        print(self.avg_sold)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model(reward['items'])
        

