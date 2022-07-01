from Learner5D import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner5D(Learner5D):

    def __init__(self, n_arms, arms):#arms MUST BE NORMALIZED
        super().__init__(n_arms)
        self.arms = np.array([arms,arms,arms,arms,arms]) #matrix for each product the arms arms[i][j] i product, j arm's index
 
        self.means = np.zeros((n_arms,n_arms,n_arms,n_arms,n_arms));#mean of the [i,j,k,m,l] multi-arm
        self.sigmas = np.ones((n_arms,n_arms,n_arms,n_arms,n_arms))*10;#var of the [i,j,k,m,l] multi-arm
        self.pulled_arms = [[]]#the zeroth position occupied by the optimum found by the algorithm
        self.collected_rewards=np.append(self.collected_rewards,0)#the zeroth position occupied by the optimum found by the algorithm
        alpha = 1 # 10 in prof code
        kernel = C(1.0, constant_value_bounds="fixed")*RBF(0.25, length_scale_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, n_restarts_optimizer=10)
        
        

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        i=pulled_arm[0];
        j=pulled_arm[1];
        k=pulled_arm[2];
        l=pulled_arm[3];
        m=pulled_arm[4];
        self.pulled_arms.append([self.arms[0][i],self.arms[1][j],self.arms[2][k],self.arms[3][l],self.arms[4][m]])
        #so that in pulled arms there are 5D points

    def update_model(self,optimal_arm,optimal_reward):
        optimal_arm=[self.arms[0][optimal_arm[0]],self.arms[1][optimal_arm[1]],self.arms[2][optimal_arm[2]],self.arms[3][optimal_arm[3]],self.arms[4][optimal_arm[4]]]
        self.pulled_arms[0]=optimal_arm
        self.collected_rewards[0]=optimal_reward
        x = self.pulled_arms
        y = self.collected_rewards*6/optimal_reward#the 2 is a value which distance the neighboor of the optimal to the other superarms
        self.gp.fit(x, y.reshape(-1,1))
        #Brute force with double cycle to generate matrix, there is definetly a better way
        for i in range(self.n_arms):
            for j in range(self.n_arms):
                for k in range(self.n_arms):
                    for l in range(self.n_arms):
                        for m in range(self.n_arms):
                            input_pred=np.array([[self.arms[0][i],self.arms[1][j],self.arms[2][k],self.arms[3][l],self.arms[4][m]]])
                            self.means[i,j,k,l,m], self.sigmas[i,j,k,l,m] = self.gp.predict(input_pred, return_std = True)
        
        #means_array, sigmas_array = self.gp.predict(self.lattice_samples_features, return_std = True)
        #sigmas_array=np.reshape(sigmas_array,(len(self.arms[0]),len(self.arms[1]),len(self.arms[2]),len(self.arms[3]),len(self.arms[4])))
        #self.sigmas = np.maximum(sigmas_array, 1e-2)
        #self.means=np.reshape(means_array,(len(self.arms[0]),len(self.arms[1]),len(self.arms[2]),len(self.arms[3]),len(self.arms[4])))
        

    def update(self, pulled_arm, reward):
        #THE REWARD MUST BE NORMALIZED!!
        self.t += 1
        self.update_observations(pulled_arm, reward)
        

    def pull_arm(self,optimal_arm, optimal_reward,constraint_mask):
# constraint_mask a list of tuple of length 5 which identify the unavailable multi_arms
        self.update_model(optimal_arm, optimal_reward)
        sampled_values = np.random.normal(self.means, self.sigmas)
        for t in constraint_mask:
            sampled_values[t[0],t[1],t[2],t[3],t[4]]=-np.inf;
        idx=np.unravel_index(np.argmax(sampled_values),sampled_values.shape)
        return idx
