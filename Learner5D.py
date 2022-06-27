import numpy as np

class Learner5D:
    def __init__(self,n_arms):
        self.n_arms=n_arms#per product
        self.t=0
        self.rewards_per_arm= x = [[]for i in range(n_arms**5)]
        #is the matrix spiattellata in a vector: M[i,j,k,l,m] identify the multiarm
        #with i-th budget on first product, j-th on second, k-th on third and so on
        #the mapping from matrix to vector is M[i,j,k,l,m]=v[i+n_arms*j+n_arms^2*k+n_arms^3*l+n_arms^4*m]
        self.collected_rewards=np.array([])

    def update_observations(self,pulled_arm,reward):#!Attention Reward must include also the eventual budget not used
        #pulled_arm vector of length 5, contains the arm pulled for each product
        i=pulled_arm[0];
        j=pulled_arm[1];
        k=pulled_arm[2];
        l=pulled_arm[3];
        m=pulled_arm[4];
        
        self.rewards_per_arm[i+self.n_arms*j+pow(self.n_arms,2)*k+pow(self.n_arms,3)*l+pow(self.n_arms,4)*m].append(reward)
        self.collected_rewards=np.append(self.collected_rewards,reward)
