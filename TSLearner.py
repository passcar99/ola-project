from audioop import avg
from random import sample
from Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import List, Dict
from tqdm import tqdm
from scipy import stats
from Algorithms import budget_allocations
from Environment import Environment
from RandomEnvironment import RandomEnvironment



class GPTS_Learner(Learner):

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins,environment_type = 'fast'):
        super().__init__(len(arms),conpam_matrix,con_matrix, prob_buy, avg_sold, margins)
        self.means = np.zeros(self.n_products, self.n_arms)
        self.sigmas = np.ones(self.n_products, self.n_arms)*10
        self.pulled_arms = [[] for _ in self.n_products]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.gps = []
        for _ in self.n_products:
            alpha = 1 # 10 in prof code
            kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))
            self.gp.append(GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=10))
        if environment_type == 'fast':
            self.env = Environment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)
        else:
            self.env = RandomEnvironment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)


    def update_observations(self, pulled_arms, reward):
        super().update_observations(pulled_arms, reward)
        for product in range(self.n_products):
            self.pulled_arms[product].append(pulled_arms[product])
            self.rewards_per_product[product].append(reward[product+1])

    def update_model(self):
        for product in range(self.n_products):
            x = np.atleast_2d(self.pulled_arms[product]).T
            y = self.rewards_per_product[product]
            self.gp[product].fit(x, y)
            self.means[product], self.sigmas[product] = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
            self.sigmas[product] = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        expected_margins = np.array((self.n_products))
        for p in range(self.n_products):
            expected_margins = np.sum(self.env.simplified_round(p, n_sim = 100))
        value_matrix = sampled_values * expected_margins
        budget_allocations(value_matrix, self.arms, subtract_budget=True)
        return np.argmax(sampled_values, axis=1)



if __name__ == '__main__':
    
