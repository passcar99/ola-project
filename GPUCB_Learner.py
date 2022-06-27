from Learner import*
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import List, Dict
from tqdm import tqdm
from scipy import stats
from audioop import avg
from random import sample
from Algorithms import budget_allocations
from Environment import Environment
from RandomEnvironment import RandomEnvironment

class GPUCB_Learner(Learner):
    
     def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds,environment_type = 'fast'):
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, avg_sold, margins, bounds)
        self.means = np.zeros((self.n_products, self.n_arms))
        #self.sigmas = np.ones((self.n_products, self.n_arms))*10
        self.confidence = np.ones((self.n_products, self.n_arms))*np.inf #init. the CBs of all the arms of all the products = inf
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.gps = []
        for _ in range(self.n_products):
            alpha = 1
            kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))
            self.gps.append(GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=10))
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
            x = np.atleast_2d(self.pulled_arms[product])
            y = self.rewards_per_product[product]
            self.gps[product].fit(x, y)
            means, sigmas = self.gps[product].predict(self.arms.reshape(-1, 1), return_std = True)
            print(means)
            
            #for ucb the updating should be ?
            self.means[product,pulled_arm] = (self.means[pulled_arm]*(self.t-1)+reward)/self.t
            for a in range(self.n_arms):
                n_samples = len(self.rewards_per_arm[a]) #now n_samples is n_samples at time (t-1)
                self.confidence[product,a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
                
            #self.means[product], self.sigmas[product] = means.flatten(), sigmas.flatten()
            #self.sigmas[product] = np.maximum(self.sigmas[product], 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
    
        
    def pull_arm(self):
        expected_margins = np.array((self.n_products))
        for p in range(self.n_products):
            ucb = self.means[p] + self.confidence[p] #where should i use it?
            expected_margins = np.sum(self.env.simplified_round(p, n_sim = 100))
            #return np.random.choice(np.where(ucb == ucb.max())[0]) -> how check this?
         value_matrix = sampled_values * expected_margins
        return budget_allocations(value_matrix, self.arms, subtract_budget=True)[0]

   
    
