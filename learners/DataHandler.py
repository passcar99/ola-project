from .Learner import Learner
from typing import List, Dict
import numpy as np


class UserDataContext:
    def __init__(self, features, n_users, items_sold) -> None:
        self.features = features # like [0, 0]
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.n_users = n_users
        self.items_sold = items_sold



class Context:
    # list of features like [[0, 0], [0, 1]]
    def __init__(self, feature_list) -> None:
        self.feature_list  = feature_list

class DataHandler(Learner):

    NAME = "DataHandler"

    def __init__(self, arms, conpam_matrix: List[Dict], con_matrix, prob_buy, margins, bounds, n_features):
        avg_sold = np.ones((len(con_matrix))) # dummy number 
        #TODO get more environments for each context
        super().__init__(arms, conpam_matrix, con_matrix, prob_buy, avg_sold, margins, bounds)
        self.avg_sold = []*(2**n_features)

    
        
    def update(self, pulled_arms, reward):
        pass

    def pull_arm(self):
        pass
        
