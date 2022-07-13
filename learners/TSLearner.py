from .Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from typing import List, Dict
from environment.Algorithms import budget_allocations
from environment.Environment import Environment
from environment.RandomEnvironment import RandomEnvironment



class GPTS_Learner(Learner):
    """ 
    Gaussian Process Thompson Sampling bandit. It estimated the click rate(?) of each product independently.
    At every round it computes the alphas and the expected margins and fills the DP table. Then selects one of the 
    feasible superarm.
    
    """
    NAME = "GPTS_Learner"
    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds,environment_type = 'fast', sliding_window=1000000):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: data about the environment (see the environment classes).
        :param con_matrix: connectivity matrix of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param avg_sold: average quantity of items bought for each product when displayed as primary.
        :param margins: margin (profit) for each arm.
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, avg_sold, margins, bounds)
        self.means = np.zeros((self.n_products, self.n_arms))
        self.sigmas = np.ones((self.n_products, self.n_arms))*10
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.sliding_window=sliding_window
        self.gps = []
        for _ in range(self.n_products):
            alpha = 1e-5 # 10 in prof code
            kernel = C(1.0, (1e-5, 1e3))*RBF(1.0, (1e-5, 1e3)) + W(1.0)
            self.gps.append(
                GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=10, copy_X_train=False
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
        """ 
        Update the Gaussian Processes for every product to incorporate the new data.
        """
        for product in range(self.n_products):
            x = np.atleast_2d(self.pulled_arms[product][-self.sliding_window:])
            y = self.rewards_per_product[product][-self.sliding_window:]
            gp = self.gps[product]
            gp.fit(x, y)
            means, sigmas = gp.predict(self.arms.reshape(-1, 1), return_std = True)
            self.means[product], self.sigmas[product] = means.flatten(), sigmas.flatten()
            self.sigmas[product] = np.maximum(self.sigmas[product], 1e-5)
            

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        """ 
        Sample the values of alphas and fill the DP table with alpha*margin*avg_n_users. Then run the bidding algorithm
        and select the optimal superarm.
        """
        value_matrix = self.compute_value_matrix()
        return budget_allocations(value_matrix, self.arms, subtract_budget=True)[0]
        
    def compute_value_matrix(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        value_matrix = np.zeros((self.n_products, self.n_arms))
        for p in range(self.n_products):
            expected_margin = self.env.simplified_round(p, n_sim = 1000)
            value_matrix[p, :] = sampled_values[p, :]* expected_margin * self.avg_n_users
            value_matrix[p, self.unfeasible_arms[p]] = -np.inf
        return value_matrix
        
    def pull_arm(self):
        value_matrix = self.compute_value_matrix()
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
    arms = np.array([0, 5, 10, 15, 20, 25])
    bounds = np.array([[-1, 100],[-1, 100],[-1, 100],[-1, 100],[-1, 100]])
    learner = GPTS_Learner(arms, conpam_matrix, connectivity_matrix, prob_buy, avg_sold, margins, bounds ,'fast')

    for _ in range(100):
        arm = learner.pull_arm()
        print(arm)
        feedback = env.round(arm.flatten())

        learner.update(arm, feedback[0])
        #TODO

