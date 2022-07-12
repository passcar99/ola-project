from .Learner import Learner
from typing import List, Dict
import numpy as np
from environment.Environment import Environment
from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner import GPTS_Learner
from learners.GPUCB_Learner import GPUCB_Learner


class UserDataContext:
    def __init__(self, features, n_users, items_sold) -> None:
        self.features = features # like [0, 0]
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.n_users = n_users
        self.items_sold = items_sold


class Context:
    # list of features like [[0, 0], [0, 1]]
    def __init__(self, feature_list, learner) -> None:
        self.feature_list  = feature_list
        self.learner = learner

class ContextManager(Learner):

    NAME = "ContextManager"

    def __init__(self, arms, conpam_matrix: List[Dict], con_matrix, prob_buy, margins, bounds, feature_list, environment_type = 'fast', learner_type = "TS"):
        self.avg_sold = np.ones((len(con_matrix))) # dummy number 
        #TODO get more environments for each context
        super().__init__(arms, conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins, bounds)
        if environment_type == 'fast':
            self.env = Environment(conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins)
        else:
            self.env = RandomEnvironment(conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins)
        self.user_data_contexts = {}
        feature_list = []
        for i in range(2):
            for j in range(2):
                feature_list.append([i, j])
                self.user_data_contexts[feature_list[-1]] = UserDataContext(feature_list[-1], 100, self.avg_sold)
        self.learner_type = learner_type
        if self.learner_type == "TS":
            learner = GPTS_Learner(self.arms, conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins, bounds) 
        elif self.learner_type=="UCB":
            learner = GPUCB_Learner(self.arms, conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins, bounds) 
        else:
            raise Exception()
        self.contexts = [Context(feature_list, learner)]

    
    def update_observations(self, pulled_arms, reward):
        super().update_observations(pulled_arms, reward)
        for r in reward:
            context = reward["features"]
            """ learner.update
            self.pulled_arms = [[] for _ in range(self.n_products)]
            self.rewards_per_product = [[] for _ in range(self.n_products)]
            self.n_users = n_users
            self.items_sold = items_sold
            self.user_data_contexts[reward.features]. """
        
    def update_model(self):
        if self.t%14==0: # every two weeks
            pass # run context generation

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.update_model()

    def pull_arm(self):
        # take estimated matrix from each learner and run algorithm
        pass
        
