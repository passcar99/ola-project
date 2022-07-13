import numpy as np
from scipy import stats
from copy import deepcopy
from .UserCategory import UserCategory

class Environment():
    """ conpam_matrix list of user classes.
    Connectivity matrix of the graph(can change with seasoality?) P_ij denotes
    the probability of clicking on j-th product when displayed on i_th page, given
    that the secondary products are fixed the lambda is implicit.
    Lambda decay from being the second secondary product.
    Prob_Buy probability that i-th product is bought """
    def __init__(self,conpam_matrix,con_matrix,prob_buy,expected_number_sold,margins):
        self.user_classes = []
        for user_class in conpam_matrix:
            self.user_classes.append(UserCategory(**user_class))
        self.con_matrix=con_matrix
        self.lam=0.5#implicit in Con_matrix
        self.prob_buy=prob_buy
        self.expected_number_sold=expected_number_sold
        self.margins=margins
        self.n_prods = len(self.margins)


    """ Developed with only ONE quantity bought and all the item have same price, to include the number of item distribution must multiply wherever there is a Prob_Buy
    the expected value of object bought together with the price of such object, or in other words the thing that is missing is the multiplication of the probabilities
    by the returns associated to that probability """
    def pull_arm(self,budgets):
        #conpam_matrix will be actually be get from the functions, now for testing we will just add
        cusum = 0

        for i, user_class in enumerate(self.user_classes):
            n_users = user_class.avg_number
            alphas = stats.dirichlet.rvs(user_class.get_alpha_from_budgets(budgets), size=1)[0]
            cusum += self.round(alphas)*n_users
             
        return cusum

    def pull_arm_excpected_value(self,budgets,time):
        #conpam_matrix will be actually be get from the functions, now for testing we will just add
        cusum = 0

        for i, user_class in enumerate(self.user_classes):
            n_users = user_class.avg_number
            DirPar=user_class.get_alpha_from_budgets(budgets,time)
            ParSum=np.sum(DirPar)
            alpha_mean = [i/ParSum for i in DirPar]
            cusum += self.round(alpha_mean)*n_users
             
        return cusum

    def round(self,alphas):#the clayrvoiant algorithm will innput the mean alphas(because he know the functions)
        
        probability_from_alpha1=alphas[1]*self.prob_buy[0]*self.site_landing(0,np.ones((5,1)))
        probability_from_alpha2=alphas[2]*self.prob_buy[1]*self.site_landing(1,np.ones((5,1)))
        probability_from_alpha3=alphas[3]*self.prob_buy[2]*self.site_landing(2,np.ones((5,1)))
        probability_from_alpha4=alphas[4]*self.prob_buy[3]*self.site_landing(3,np.ones((5,1)))
        probability_from_alpha5=alphas[5]*self.prob_buy[4]*self.site_landing(4,np.ones((5,1)))

        probabilities_on_nodes=probability_from_alpha1+probability_from_alpha2+probability_from_alpha3+probability_from_alpha4+probability_from_alpha5

        value_from_node1=probabilities_on_nodes[0]*self.expected_number_sold[0]*self.margins[0]
        value_from_node2=probabilities_on_nodes[1]*self.expected_number_sold[1]*self.margins[1]
        value_from_node3=probabilities_on_nodes[2]*self.expected_number_sold[2]*self.margins[2]
        value_from_node4=probabilities_on_nodes[3]*self.expected_number_sold[3]*self.margins[3]
        value_from_node5=probabilities_on_nodes[4]*self.expected_number_sold[4]*self.margins[4]        

        total_value=value_from_node1+value_from_node2+value_from_node3+value_from_node4+value_from_node5
        return total_value

    def simplified_round(self, product, n_sim = 0):
        alphas = np.zeros((len(self.con_matrix)+1))
        alphas[product+1 ]= 1
        return self.round(alphas)

    def alpha_functions(self, phase=None):
        alpha_functions = []
        if phase is not None: 
            for user_cat in self.user_classes:
                alpha_functions.append(user_cat.alpha_functions[phase])
        else:
            for user_cat in self.user_classes:
                alpha_functions.append(user_cat.alpha_functions)
        return alpha_functions



    def site_landing(self,landing_product,activated_nodes):
        ret=np.zeros(5)
        if(activated_nodes[landing_product] == 0):
            return 0        
        ret[landing_product]=1#return always 1 for the current node
        #exctract landing product column
        connectedness=self.con_matrix[landing_product]#extract landing_product row
        connectedness=(activated_nodes.T*connectedness)[0]#available connections
        if np.sum(connectedness)==0:
            return ret#only current node returna
        activated_nodes[landing_product]=0#deactivate current node for the following steps
        #sec_prod=np.nonzero(Connectedness)[0]#secondary products
        sec_prod = np.flip(np.argsort(connectedness)[-2:])
        if sec_prod.size==2:
            #values from getting ONLY to first or second secondary
            first_ret=self.prob_buy[sec_prod[0]]*connectedness[sec_prod[0]]*(1-connectedness[sec_prod[1]])*self.site_landing(sec_prod[0],deepcopy(activated_nodes))
            second_ret=self.prob_buy[sec_prod[1]]*connectedness[sec_prod[1]]*(1-connectedness[sec_prod[0]])*self.site_landing(sec_prod[1],deepcopy(activated_nodes))
            #case in which both are visited
            #both visited, first product
            activated_nodes[sec_prod[1]]=0
            both1_ret=self.prob_buy[sec_prod[0]]*connectedness[sec_prod[1]]*connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],deepcopy(activated_nodes))
            activated_nodes[sec_prod[1]]=1
            activated_nodes[sec_prod[0]]=0
            both2_ret=self.prob_buy[sec_prod[1]]*connectedness[sec_prod[1]]*connectedness[sec_prod[0]]*self.site_landing(sec_prod[1],deepcopy(activated_nodes))
            #can be executed recoursively
            return ret+first_ret+second_ret+both1_ret+both2_ret
            
        first_ret=self.prob_buy[sec_prod[0]]*connectedness[sec_prod[0]]*self.site_landing(sec_prod[0],deepcopy(activated_nodes))
        return ret+first_ret

        

