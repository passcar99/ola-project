import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from copy import deepcopy

class Environment():
    #conpam_matrix with in  i-th column the i-th concentration parameter of the
    #dirichlet distribution, in the j-th row the j-th seasonality period.
    #Connectivity matrix of the graph(can change with seasoality?) P_ij denotes
    #the probability of clicking on j-th product when displayed on i_th page, given
    #that the secondary products are fixed the lambda is implicit.
    #Lambda decay from being the second secondary product.
    #Prob_Buy probability that i-th product is bought
    def __init__(self,conpam_matrix,Con_matrix,Prob_Buy,Expected_number_sold,Margins):
        self.conpam_matrix=conpam_matrix;
        self.Con_matrix=Con_matrix;
        self.lam=0.5;#implicit in Con_matrix
        self.Prob_Buy=Prob_Buy
        self.Expected_number_sold=Expected_number_sold;
        self.Margins=Margins;


    #Developed with only ONE quantity bought and all the item have same price, to include the number of item distribution must multiply wherever there is a Prob_Buy
    #the expected value of object bought together with the price of such object, or in other words the thing that is missing is the multiplication of the probabilities
    #by the returns associated to that probability
    def round(self):
        alphas=stats.dirichlet.rvs(self.conpam_matrix[0], size=1, random_state=42)[0];
        alphas[0];#to competitors
        
        Probability_of_alpha1=alphas[1]*self.Prob_Buy[0]*self.site_landing(0,np.ones((5,1)));
        Probability_of_alpha2=alphas[2]*self.Prob_Buy[1]*self.site_landing(1,np.ones((5,1)));
        Probability_of_alpha3=alphas[3]*self.Prob_Buy[2]*self.site_landing(2,np.ones((5,1)));
        Probability_of_alpha4=alphas[4]*self.Prob_Buy[3]*self.site_landing(3,np.ones((5,1)));
        Probability_of_alpha5=alphas[5]*self.Prob_Buy[4]*self.site_landing(4,np.ones((5,1)));

        Value_from_alpha1=Probability_of_alpha1*self.Expected_number_sold[1]*self.Margins[1];
        Value_from_alpha2=Probability_of_alpha2*self.Expected_number_sold[2]*self.Margins[2];
        Value_from_alpha3=Probability_of_alpha3*self.Expected_number_sold[3]*self.Margins[3];
        Value_from_alpha4=Probability_of_alpha4*self.Expected_number_sold[4]*self.Margins[4];
        Value_from_alpha5=Probability_of_alpha5*self.Expected_number_sold[5]*self.Margins[5];
        

        Total_Value=Value_from_alpha1+Value_from_alpha2+Value_from_alpha3+Value_from_alpha4+Value_from_alpha5;

        return Total_Value;

    def alpha_function(self, min_budget, max_budget, alpha_bar): #assuming linear behaviour. TODO check
        x1, y1 = min_budget, 0
        x2, y2 = max_budget, alpha_bar
        #return a function to be called as f(inputs) where inputs can be a number or an array
        return interp1d([x1, x2], [y1, y2], kind='linear', bounds_error=False, fill_value=(y1, y2) )


    def site_landing(self,landing_product,activated_nodes):
        ret=np.zeros(5);
        if(activated_nodes[landing_product] == 0):
            return 0        
        ret[landing_product]=1;#return always 1 for the current node
        #exctract landing product column
        Connectedness=self.Con_matrix[landing_product];#extract landing_product row
        Connectedness=(activated_nodes.T*Connectedness)[0];#available connections
        if np.sum(Connectedness)==0:
            return ret;#only current node returna
        activated_nodes[landing_product]=0;#deactivate current node for the following steps
        #sec_prod=np.nonzero(Connectedness)[0];#secondary products
        sec_prod = np.flip(np.argsort(Connectedness)[-2:])
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

        


