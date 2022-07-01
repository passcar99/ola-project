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
from GPTS_Learner import GPTS_Learner5D



class GPTS_Learner_TOP5D(Learner):

    def __init__(self, arms, conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins, bounds,environment_type = 'fast'):
        super().__init__(arms,conpam_matrix,con_matrix, prob_buy, avg_sold, margins, bounds)
        self.means = np.zeros((self.n_products, self.n_arms))
        self.sigmas = np.ones((self.n_products, self.n_arms))*10
        self.pulled_arms = [[] for _ in range(self.n_products)]
        self.rewards_per_product = [[] for _ in range(self.n_products)]
        self.gps = []
        #Feasibility mask computation--
        self.constraint_mask=[]
        for m in range(len(arms)):
            for l in range(len(arms)):
                for k in range(len(arms)):
                    for j in range(len(arms)):
                        for i in range(len(arms)):
                            if (arms[i]+arms[j]+arms[k]+arms[l]+arms[m]>arms[-1] 
                            or arms[i]<bounds[0][0] 
                            or arms[i]>bounds[0][1]
                            or arms[j]<bounds[1][0] 
                            or arms[j]>bounds[1][1]
                            or arms[k]<bounds[2][0] 
                            or arms[k]>bounds[2][1]
                            or arms[l]<bounds[3][0] 
                            or arms[l]>bounds[3][1]
                            or arms[m]<bounds[4][0] 
                            or arms[m]>bounds[4][1]
                            ):
                                self.constraint_mask.append([i,j,k,l,m])
        #------------------------------
        #GP5--
        self.GP5 = GPTS_Learner5D(len(arms),(arms-min(arms))/max(arms))
        #-----
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
            """ if self.t >= 10:
                plt.plot(self.arms,means)
                plt.fill_between(self.arms, means-sigmas, means+sigmas)
                plt.show() """
            self.means[product], self.sigmas[product] = means.flatten(), sigmas.flatten()
            self.sigmas[product] = np.maximum(self.sigmas[product], 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        profits = reward['profit']
        arms=list(self.arms)
        pulled_arm_idx=[arms.index(pulled_arm[0]),arms.index(pulled_arm[1]),arms.index(pulled_arm[2]),arms.index(pulled_arm[3]),arms.index(pulled_arm[4])]
        self.GP5.update(pulled_arm_idx,profits)

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        value_matrix = np.zeros((self.n_products, self.n_arms))
        for p in range(self.n_products):
            expected_margin = self.env.simplified_round(p, n_sim = 1000)
            value_matrix[p, :] = sampled_values[p, :]* expected_margin * self.avg_n_users
            value_matrix[p, self.unfeasible_arms[p]] = -np.inf
        
        optimal_arm, optimal_reward=budget_allocations(value_matrix, self.arms, subtract_budget=True)
        arms=list(self.arms)
        optimal_arm_idx=[arms.index(optimal_arm[0]),arms.index(optimal_arm[1]),arms.index(optimal_arm[2]),arms.index(optimal_arm[3]),arms.index(optimal_arm[4])]
        pull_arm_idx=self.GP5.pull_arm(optimal_arm_idx,optimal_reward,self.constraint_mask)
        #return np.array([ [arms[pull_arm_idx[0]]],[arms[pull_arm_idx[1]]],[arms[pull_arm_idx[2]]],[arms[pull_arm_idx[3]]],[arms[pull_arm_idx[4]]]])
        ret=np.array([ [arms[pull_arm_idx[0]]],[arms[pull_arm_idx[1]]],[arms[pull_arm_idx[2]]],[arms[pull_arm_idx[3]]],[arms[pull_arm_idx[4]]]])
        print('*******************')
        print(ret==optimal_arm)
        print('*******************')
        return ret
        



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

