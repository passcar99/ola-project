import numpy as np
import scipy as sc

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

    def round(self):
            return 0;



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
                First_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[0]]*(1-Connectedness[sec_prod[1]])*self.site_landing(sec_prod[0],activated_nodes);
                Second_ret=self.Prob_Buy[sec_prod[1]]*Connectedness[sec_prod[1]]*(1-Connectedness[sec_prod[0]])*self.site_landing(sec_prod[1],activated_nodes);
                #case in which both are visited
                #both visited, first product
                activated_nodes[sec_prod[1]]=0;
                Both1_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[1]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],activated_nodes);
                activated_nodes[sec_prod[1]]=1;
                activated_nodes[sec_prod[0]]=0;
                Both2_ret=self.Prob_Buy[sec_prod[1]]*Connectedness[sec_prod[1]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[1],activated_nodes);
                #can be executed recoursively
                return ret+First_ret+Second_ret+Both1_ret+Both2_ret;
            
            First_ret=self.Prob_Buy[sec_prod[0]]*Connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],activated_nodes);
            return ret+First_ret;
        


        


