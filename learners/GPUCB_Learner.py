from .Learner import*
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from typing import List, Dict
from environment.Algorithms import budget_allocations
from environment.Environment import Environment
from environment.RandomEnvironment import RandomEnvironment
import math



class GPUCB_Learner(Learner):
    
    NAME = "GPUCB_Learner"

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds,environment_type = 'fast',method='normal',sliding_window=1000000, bound_type='gaussian'):
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, avg_sold, margins, bounds)
        self.means = np.zeros((self.n_products, self.n_arms))
        self.sigmas = np.ones((self.n_products, self.n_arms))*10
        #self.confidence = np.ones((self.n_products, self.n_arms))*np.inf #init. the CBs of all the arms of all the products = inf
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.sliding_window=sliding_window#sliding window
        self.last_change=1#change detection
        self.method = method
        self.gps = []
        for _ in range(self.n_products):
            alpha = 1e-5 # 10 in prof code
            kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3)) + W(1.0)
            self.gps.append(
                GaussianProcessRegressor(
                    kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=10, copy_X_train=False
                    ) # keep a reference to training data to avoid copying it every time
                ) 
        if environment_type == 'fast':
            self.env = Environment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)
        else:
            self.env = RandomEnvironment(conpam_matrix, con_matrix, prob_buy, avg_sold, margins)
        self.bound_type = bound_type
            
    def update_observations(self, pulled_arms, reward):
        super().update_observations(pulled_arms, reward)
        alphas = reward['alphas']
        Standard5DNoarmal_Sample=np.array([])
        for product in range(self.n_products):
            self.pulled_arms[product].append(pulled_arms[product])
            self.rewards_per_product[product].append(alphas[product+1])
            #change detection
            arms=list(self.arms)
            pulled_arm_idx=arms.index(pulled_arms[product])
            Standard5DNoarmal_Sample=np.append(Standard5DNoarmal_Sample,(alphas[product+1]-self.means[product][pulled_arm_idx])/self.sigmas[product][pulled_arm_idx])
            #if (abs((alphas[product+1]-self.means[product][pulled_arm_idx]))/self.sigmas[product][pulled_arm_idx]>1.95 and self.method=="detect"):#1.95 is Z_95 2.57 is circa Z_99 and 0.99^5=(circa)0.95
            #    self.last_change=len(self.pulled_arms[product])
            #    print("#########OOOLD Change detected at T="+str(self.last_change))
        Standardized_5D_sample_norm = np.linalg.norm(Standard5DNoarmal_Sample)
        #print("Standardized 5D sample norm "+ self.method +": "+str(Standardized_5D_sample_norm))
        if (Standardized_5D_sample_norm>3.88411 and self.method=="detect" and (len(self.pulled_arms[product])-self.last_change)>=10):#1.95 is Z_95 2.57 is circa Z_99 and 0.99^5=(circa)0.95
                self.last_change=len(self.pulled_arms[product])
                print("#########Change detected at T="+str(self.last_change)+" Norm Standardized 5D Sample:"+str(Standardized_5D_sample_norm))

    def update_model(self):
        for product in range(self.n_products):
            if self.method=="slide":
                x = np.atleast_2d(self.pulled_arms[product][-self.sliding_window:]) #sliding window
                y = self.rewards_per_product[product][-self.sliding_window:] #sliding window
            elif self.method=="detect":
                x = np.atleast_2d(self.pulled_arms[product][self.last_change-1:]) #change detection
                y = self.rewards_per_product[product][self.last_change-1:] #change detection
            else:
                x = np.atleast_2d(self.pulled_arms[product]) #change detection
                y = self.rewards_per_product[product] #change detection
            gp = self.gps[product]
            gp.fit(x, y)
            means, sigmas = gp.predict(self.arms.reshape(-1, 1), return_std = True)
            self.means[product], self.sigmas[product] = means.flatten(), sigmas.flatten()
            self.sigmas[product] = np.maximum(self.sigmas[product], 1e-2)
            
            if self.bound_type == "paper":
                self.sigmas *= np.sqrt(2*np.log(self.n_arms*self.t**2*np.pi**2/(6*0.01)))
            #for 2° method of ucb 
            #self.means[product,pulled_arm] = (self.means[pulled_arm]*(self.t-1)+reward)/self.t
            #for a in range(self.n_arms):
            #    n_samples = len(self.rewards_per_arm[a]) #now n_samples is n_samples at time (t-1)
            #    self.sigmas[product,a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples>0 else np.inf
         

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        
    def compute_value_matrix(self):
        upper_confidence_bounds = self.means + self.sigmas # first method 
        value_matrix = np.zeros((self.n_products, self.n_arms))
        for p in range(self.n_products):
            expected_margin = self.env.simplified_round(p, n_sim = 1000)
            value_matrix[p, :] = upper_confidence_bounds[p, :]* expected_margin * self.avg_n_users
            value_matrix[p, self.unfeasible_arms[p]] = -np.inf
        return value_matrix
        
    def pull_arm(self):
        value_matrix = self.compute_value_matrix()
        return budget_allocations(value_matrix, self.arms, subtract_budget=True)[0]



