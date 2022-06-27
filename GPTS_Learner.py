from Learner5D import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner(Learner5D):

    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms #matrix for each product the arms arms[i][j] i product, j arm's index
 
        self.means = np.zeros((n_arms,n_arms,n_arms,n_arms,n_arms));#mean of the [i,j,k,m,l] multi-arm
        self.sigmas = np.ones((n_arms,n_arms,n_arms,n_arms,n_arms))*10;#var of the [i,j,k,m,l] multi-arm
        self.pulled_arms = []
        alpha = 1 # 10 in prof code
        kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2, normalize_y=True, n_restarts_optimizer=10)
        #[N1,N2,N3,N4,N5]=np.meshgrid(arms[0][:],arms[1][:],arms[2][:],arms[3][:],arms[4][:])
        #N1=N1.flatten()
        #N2=N2.flatten()
        #N3=N3.flatten()
        #N4=N4.flatten()
        #N5=N5.flatten()
        #self.lattice_samples_features=np.array([N1,N2,N3,N4,N5]).T #matrix Mij with i identifying a superarm and j a product of such arm
        

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        i=pulled_arm[0];
        j=pulled_arm[1];
        k=pulled_arm[2];
        l=pulled_arm[3];
        m=pulled_arm[4];
        self.pulled_arms.append([self.arms[0][i],self.arms[1][j],self.arms[2][k],self.arms[3][l],self.arms[4][m]])
        #so that in pulled arms there are 5D points

    def update_model(self):
        x = self.pulled_arms
        y = self.collected_rewards
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
        reward=reward/3
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self,constraint_mask):
# constraint_mask a list of tuple of length 5 which identify the unavailable multi_arms
        sampled_values = np.random.normal(self.means, self.sigmas)
        for t in constraint_mask:
            sampled_values[t[0],t[1],t[2],t[3],t[4]]=-100000;
        idx=np.unravel_index(np.argmax(sampled_values),sampled_values.shape)
        return idx
