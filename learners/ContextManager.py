from environment.Algorithms import budget_allocations
from .Learner import Learner
from typing import List, Dict
import numpy as np
from environment.Environment import Environment
from environment.RandomEnvironment import RandomEnvironment
from learners.TSLearner import GPTS_Learner
from learners.GPUCB_Learner import GPUCB_Learner
from learners.ContextGeneration import ContextGeneration


class UserDataContext:
    def __init__(self, class_id, n_users, items_sold, context_id) -> None:
        self.class_id = class_id
        self.features = [class_id//2, class_id%2] # like [0, 0]
        self.pulled_arms = [[] for _ in range(len(items_sold))]
        self.rewards_per_product = [[] for _ in range(len(items_sold))]
        self.n_users = n_users
        self.items_sold = items_sold
        self.context_id = context_id


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
        for i in range(4):
                feature_list.append(i)
                self.user_data_contexts[i] = UserDataContext(i, 100, self.avg_sold, 0)
        if learner_type == "TS":
            self.learner_type = GPTS_Learner
        elif learner_type=="UCB":
            self.learner_type = GPUCB_Learner
        else:
            raise Exception()
        learner = self.learner_type(self.arms, conpam_matrix, con_matrix, prob_buy, self.avg_sold, margins, bounds) 
        context = Context(feature_list, learner)
        self.contexts = [context]
        self.tot_visits = 0

    
    def update_observations(self, pulled_arms, reward):
        self.collected_rewards.append(np.sum(r['profit'] for r in reward))
        rewards_per_context = [[r for r in reward if r["features"] in c.feature_list] for c in self.contexts]
        for context_id, context in enumerate(self.contexts):
            n_users = np.sum([r["n_users"] for r in rewards_per_context[context_id]])
            for r in rewards_per_context[context_id]: # update per class statistics
                user_group = r["features"]
                alphas = r["alphas"]
                user_data = self.user_data_contexts[user_group]
                for product in range(self.n_products): # TODO correct
                    user_data.pulled_arms[product].append(pulled_arms[context_id][1][product]*r["n_users"]/n_users)
                    user_data.rewards_per_product[product].append(alphas[product+1])
                user_data.n_users = (user_data.n_users * (self.t-1) + r["n_users"])/(self.t)  
                items_sold = r["items"]
                today_visits = items_sold.shape[0]

                items_sold[items_sold==0] = np.nan # do not consider 0s when taking the mean (assume at least one bought)
                today_avg_sold = np.nanmean(items_sold, axis=0)
                mask = np.logical_not(np.isnan(today_avg_sold))
                #running average to be more efficient. Only at the first iteration tot_visits==0, consequently avg_sold is ignored
                self.avg_sold[mask] = (self.avg_sold[mask] * self.tot_visits + today_avg_sold[mask]* today_visits)/(self.tot_visits + today_visits)
                self.tot_visits += today_visits
                self.env.avg_sold = self.avg_sold
            context_reward = {
                "n_users": n_users,
                "alphas": np.sum(np.array([r["n_users"]*r["alphas"] for r in rewards_per_context[context_id]]), axis=0 )/n_users,
                "profit": np.sum([r["profit"] for r in rewards_per_context[context_id]])
            }
            context.learner.env.avg_sold = self.avg_sold
            context.learner.update(np.array(pulled_arms[context_id][1]), context_reward) # IMPORTANT assume same order as played

        
    def update_model(self):
        if self.t%14==0 and self.t>14: # every two weeks after first month
            context_generation_alg = ContextGeneration(self.n_products, self.arms, self.margins, self.env, self.unfeasible_arms)
            grouped_classes = np.zeros((len(self.user_data_contexts)))
            context_generation_alg.compute_split(self.user_data_contexts.values(), [0, 1], grouped_classes)
            self.contexts = [] # discard old context structure
            print("grouped_classes: ")
            print(grouped_classes)
            for group in np.unique(grouped_classes):
                context_learner = self.learner_type(self.arms, self.conpam_matrix, self.con_matrix, self.prob_buy, self.avg_sold, self.margins, self.bounds)
                context_classes_ids = np.arange(4)[grouped_classes==group]
                self.contexts.append(
                    Context(context_classes_ids, context_learner# only classes for identified group
                     ))
                avg_n_users = 0
                for user_cat_id in context_classes_ids:
                    user_cat = self.user_data_contexts[user_cat_id]
                    user_cat.context_id = len(self.contexts)-1
                    avg_n_users += user_cat.n_users
                    for product in range(self.n_products):
                        context_learner.pulled_arms[product].extend(user_cat.pulled_arms[product])
                        context_learner.rewards_per_product[product].extend(user_cat.rewards_per_product[product])
                context_learner.avg_n_users = avg_n_users

    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms, reward)
        self.update_model()

    def pull_arm(self):
        # take estimated matrix from each learner and run algorithm
        value_matrix = np.vstack([context.learner.compute_value_matrix() for context in self.contexts])
        #user_classes = [context.feature_list for context in self.contexts]
        super_arm_shallow = budget_allocations(value_matrix, self.arms, True)[0]
        super_arm = [(context.feature_list, super_arm_shallow[i*5:i*5+self.n_products]) for i, context in enumerate(self.contexts)]
        return super_arm
        
        
