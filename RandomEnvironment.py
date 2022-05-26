import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm
from Environment import Environment
from UserClass import UserClass
from typing import List, Dict

class RandomEnvironment():
    #conpam_matrix with in  i-th column the i-th concentration parameter of the
    #dirichlet distribution, in the j-th row the j-th seasonality period.
    #Connectivity matrix of the graph(can change with seasoality?) P_ij denotes
    #the probability of clicking on j-th product when displayed on i_th page, given
    #that the secondary products are fixed the lambda is implicit.
    #Lambda decay from being the second secondary product.
    #Prob_Buy probability that i-th product is bought
    def __init__(self,conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins):
        self.user_classes = []
        for user_class in conpam_matrix:
            self.user_classes.append(UserClass(**user_class))
        self.con_matrix=con_matrix
        self.lam=0.5;#implicit in Con_matrix
        self.prob_buy=prob_buy
        self.n_sim = n_sim
        self.avg_sold = avg_sold
        self.margins = margins
        

    def round(self, budgets):
        alphas_list = []
        n_users = []
        for user_class in self.user_classes:
            n_users.append(np.random.poisson(user_class.avg_number))
            alphas_list.append(
                stats.dirichlet.rvs(user_class.get_alpha_from_budgets(budgets), size=1, random_state=42)[0])
        #alphas[0];#to competitors
        cusum = np.zeros((5, 1))
        for i, n in enumerate(n_users):
            for _ in tqdm(range(n)):
                landing_product = np.nonzero(np.random.multinomial(1, alphas_list[i]))[0][0]
                if landing_product == 0: #competitor
                    continue
                cusum += self.site_landing(landing_product-1, np.zeros((5, 1), dtype=int), np.zeros((5, 1)))
        return cusum/self.n_sim


    def alpha_function(self, min_budget, max_budget, alpha_bar): #assuming linear behaviour. TODO check
        x1, y1 = min_budget, 0
        x2, y2 = max_budget, alpha_bar
        #return a function to be called as f(inputs) where inputs can be a number or an array
        return interp1d([x1, x2], [y1, y2], kind='linear', bounds_error=False, fill_value=(y1, y2) )

    def site_landing(self,landing_product,activated_nodes, bought_nodes):#this case is with only ONE quantity bought and all the item have same price
        activated_nodes[landing_product] = 1
        buy = np.random.binomial(1, self.prob_buy[landing_product])
        if buy == 0:
            return bought_nodes
        num_bought = np.random.poisson(self.avg_sold[landing_product])
        while num_bought==0:
            num_bought = np.random.poisson(self.avg_sold[landing_product])
        bought_nodes[landing_product] = num_bought*self.margins[landing_product]
        sec_prod_prob = self.con_matrix[landing_product].flatten()
        secondary_products = np.argsort(sec_prod_prob)
        if(len(secondary_products)==0) :
            return bought_nodes
        first_sec = secondary_products[-1]
        second_sec = secondary_products[-2]
        move = np.random.binomial(1, (1-activated_nodes[first_sec])*sec_prod_prob[first_sec]) # if 0 returns always 0
        if move:
            self.site_landing(first_sec, activated_nodes, bought_nodes)
        move = np.random.binomial(1, (1-activated_nodes[second_sec])*sec_prod_prob[second_sec]) # if 0 returns always 0
        if move:
            self.site_landing(second_sec, activated_nodes, bought_nodes)
        return bought_nodes

"""Just for testing""" 
if __name__=='__main__':
    con_matrix = [[2, 10, 15, 2, 22, 13]]
    connectivity_matrix = np.array([[0, 0.2, 0.4, 0.3, 0.1],
                                    [0.5, 0, 0.1, 0.3, 0.1],
                                    [0.3, 0.2, 0, 0.1, 0.4],
                                    [0.13, 0.17, 0.30, 0, 0.4],
                                    [0.16, 0.34, 0.15, 0.25, 0],
                                    ])
    """ connectivity_matrix = np.array([[0, 0, 0.9, 0.3, 0],
                                    [0, 0, 0.1, 0, 0.9],
                                    [0.3, 0, 0, 0.8, 0],
                                    [0.13, 0, 0.7, 0, 0],
                                    [0, 0.8, 0.4, 0., 0],
                                    ]) """
    prob_buy = np.array([0.1, 0.2, 0.5, 0.9, 0.7])
    n_sim = 100000

    avg_sold = [5,6,7,8,9]
    margins = [10, 20, 30, 40, 50]
    """avg_sold = [1,1,1,1,1]
    margins = [1,1,1,1,1] """
    con_matrix = [{"alpha_params": [(0, 10, 2), (5, 10, 6),(5, 20, 10),(5, 50, 6),(5, 8, 6)], "features":[0, 0], "total_mass":64, "avg_number":100}, 
                {"alpha_params": [(0, 10, 2), (5, 10, 6),(5, 20, 10),(5, 50, 6),(5, 8, 6)], "features":[0, 1], "total_mass":64, "avg_number":200}]
    env = RandomEnvironment(con_matrix, connectivity_matrix, prob_buy, avg_sold, margins)
    """import matplotlib.pyplot as plt
    plt.plot(env.alpha_function(5, 100, 20)(np.linspace(-5, 200, 2000)))
    plt.show() """
    con_matrix = [[2, 10, 15, 2, 22, 13]]
    env2 = Environment(con_matrix, connectivity_matrix, prob_buy, [10]+avg_sold, [10]+margins)
    probs = env.round([10, 20, 6,50,45]).flatten()
    probs2 = env2.round()
    print('Random environment:', probs, 'Efficient environment: ', probs2, sep='\n')
