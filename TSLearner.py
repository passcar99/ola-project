from turtle import color
from Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import List, Dict
from Algorithms import budget_allocations
from Environment import Environment
from RandomEnvironment import RandomEnvironment
import matplotlib.pyplot as plt



class GPTS_Learner(Learner):

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds,environment_type = 'fast'):
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, avg_sold, margins, bounds)
        self.means = np.zeros((self.n_products, self.n_arms))
        self.sigmas = np.ones((self.n_products, self.n_arms))*10
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.gps = []
        for _ in range(self.n_products):
            alpha = 0.1**(1/2) # 10 in prof code
            kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))
            self.gps.append(
                GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=10, copy_X_train=False
                    ) # keep a reference to training data to avoid copying it every time
                    ) 
        if environment_type == 'fast':
            self.env = Environment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)
        else:
            self.env = RandomEnvironment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)


    def update_observations(self, pulled_arms, reward):
        super().update_observations(pulled_arms, reward)
        alphas = reward['alphas']
        for product in range(self.n_products):
            self.pulled_arms[product].append(pulled_arms[product])
            self.rewards_per_product[product].append(alphas[product+1])

    def update_model(self):
        for product in range(self.n_products):
            x = np.atleast_2d(self.pulled_arms[product])
            y = self.rewards_per_product[product]
            gp = self.gps[product]
            gp.fit(x, y)
            means, sigmas = gp.predict(self.arms.reshape(-1, 1), return_std = True)
            self.means[product], self.sigmas[product] = means.flatten(), sigmas.flatten()
            self.sigmas[product] = np.maximum(self.sigmas[product], 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        value_matrix = np.zeros((self.n_products, self.n_arms))
        for p in range(self.n_products):
            expected_margin = self.env.simplified_round(p, n_sim = 1000)
            value_matrix[p, :] = sampled_values[p, :]* expected_margin
            value_matrix[p, self.unfeasible_arms[p]] = -np.inf
        
        return budget_allocations(value_matrix, self.arms, subtract_budget=True)[0]
        



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
    arms = np.array([20, 30, 40, 50, 60])
    bounds = np.array([[5, 100],[0, 80],[0, 50],[20, 100],[0, 100]])
    learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

    for _ in range(100):
        arm = learner.pull_arm()
        print(arm)
        feedback = env.round(arm.flatten())

        learner.update(arm, feedback[0])
        #TODO

