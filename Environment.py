import numpy as np
from scipy import stats
from copy import deepcopy

class Environment():
    #conpam_matrix with in  i-th column the i-th concentration parameter of the
    #dirichlet distribution, in the j-th row the j-th seasonality period.
    #Connectivity matrix of the graph(can change with seasoality?) P_ij denotes
    #the probability of clicking on j-th product when displayed on i_th page, given
    #that the secondary products are fixed the lambda is implicit.
    #Lambda decay from being the second secondary product.
    #Prob_Buy probability that i-th product is bought
    def __init__(self,conpam_matrix,Con_matrix,Prob_Buy):
        self.conpam_matrix=conpam_matrix;
        self.Con_matrix=Con_matrix;
        self.lam=0.5;#implicit in Con_matrix
        self.Prob_Buy=Prob_Buy


    #Developed with only ONE quantity bought and all the item have same price, to include the number of item distribution must multiply wherever there is a Prob_Buy
    #the expected value of object bought together with the price of such object, or in other words the thing that is missing is the multiplication of the probabilities
    #by the returns associated to that probability
    def round(self):
        alphas=stats.dirichlet.rvs(self.conpam_matrix[0], size=1)[0];
        alphas[0];#to competitors
        
        Value_from_alpha1=alphas[1]*self.Prob_Buy[0]*self.site_landing(0,np.ones((5,1)));
        Value_from_alpha2=alphas[2]*self.Prob_Buy[1]*self.site_landing(1,np.ones((5,1)));
        Value_from_alpha3=alphas[3]*self.Prob_Buy[2]*self.site_landing(2,np.ones((5,1)));
        Value_from_alpha4=alphas[4]*self.Prob_Buy[3]*self.site_landing(3,np.ones((5,1)));
        Value_from_alpha5=alphas[5]*self.Prob_Buy[4]*self.site_landing(4,np.ones((5,1)));

        Total_Value=Value_from_alpha1+Value_from_alpha2+Value_from_alpha3+Value_from_alpha4+Value_from_alpha5;

        return Total_Value;



    def site_landing(self,landing_product,activated_nodes):
        ret=np.zeros(5);
        ret[landing_product]=self.Prob_Buy[landing_product];#return always 1 for the current node
        #exctract landing product column
        Connectedness=self.Con_matrix[landing_product];#extract landing_product row
            Connectedness=(activated_nodes.T*Connectedness)[0];#available connections
        if np.sum(Connectedness)==0:
            return ret;#only current node returna
        sec_prod=np.nonzero(Connectedness)[0];#secondary products
        activated_nodes[landing_product]=0;#deactivate current node for the following steps
        
        if sec_prod.size==2:
            #values from getting ONLY to first or second secondary
            First_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[0]]*(1-Connectedness[sec_prod[1]])*self.site_landing(sec_prod[0],deepcopy(activated_nodes));
            Second_ret=self.Prob_Buy[sec_prod[1]]*Connectedness[sec_prod[1]]*(1-Connectedness[sec_prod[0]])*self.site_landing(sec_prod[1],deepcopy(activated_nodes));
            #case in which both are visited
            #both visited, first product
            activated_nodes[sec_prod[1]]=0;
            Both1_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[1]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],deepcopy(activated_nodes));
            activated_nodes[sec_prod[1]]=1;
            activated_nodes[sec_prod[0]]=0;
            Both2_ret=self.Prob_Buy[sec_prod[1]]*Connectedness[sec_prod[1]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[1],deepcopy(activated_nodes));
            #can be executed recoursively
            return ret+First_ret+Second_ret+Both1_ret+Both2_ret;
            
        First_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],deepcopy(activated_nodes));
        return ret+First_ret;

        


