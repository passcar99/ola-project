from .Learner5D import *

class TS_Learner5D(Learner5D):

    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.beta_parameters=np.ones((n_arms,n_arms,n_arms,n_arms,n_arms,2))

    def pull_arm(self,constraint_mask):# constraint_mask a list of tuple of length 5 which identify the unavailable multi_arms
        beta_outcome=np.random.beta(self.beta_parameters[:,:,:,:,:,0],self.beta_parameters[:,:,:,:,:,1])
        for t in constraint_mask:
            beta_outcome[t[0],t[1],t[2],t[3],t[4]]=-100000;
        idx=np.unravel_index(np.argmax(beta_outcome),beta_outcome.shape)
        return idx#tuple of the 5 inndexes of the pulled multi arm

    def update(self,pulled_arm,reward):#!Attention Reward must include also the eventual budget not used
        self.t+=1
        self.update_observations(pulled_arm,reward)
        i=pulled_arm[0];
        j=pulled_arm[1];
        k=pulled_arm[2];
        l=pulled_arm[3];
        m=pulled_arm[4];
        #for the update reward must be normalized...HOW?? For test taken empirically
        self.beta_parameters[i,j,k,l,m,0]=self.beta_parameters[i,j,k,l,m,0]+reward/3
        self.beta_parameters[i,j,k,l,m,1]=self.beta_parameters[i,j,k,l,m,1]+1.0-reward/3
        
