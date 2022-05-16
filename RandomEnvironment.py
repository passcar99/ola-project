import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm
from Environment import Environment

class RandomEnvironment():
    #conpam_matrix with in  i-th column the i-th concentration parameter of the
    #dirichlet distribution, in the j-th row the j-th seasonality period.
    #Connectivity matrix of the graph(can change with seasoality?) P_ij denotes
    #the probability of clicking on j-th product when displayed on i_th page, given
    #that the secondary products are fixed the lambda is implicit.
    #Lambda decay from being the second secondary product.
    #Prob_Buy probability that i-th product is bought
    def __init__(self,conpam_matrix,con_matrix, prob_buy, n_sim):
        self.conpam_matrix=conpam_matrix
        self.con_matrix=con_matrix
        self.lam=0.5;#implicit in Con_matrix
        self.prob_buy=prob_buy
        self.n_sim = n_sim

    def round(self):
        alphas=stats.dirichlet.rvs(self.conpam_matrix[0], size=1)[0]
        #alphas[0];#to competitors
        print(alphas)
        prob = np.zeros((5, 1), dtype=int)
        for _ in tqdm(range(self.n_sim)):
            landing_product = np.nonzero(np.random.multinomial(1, alphas))[0]
            if landing_product == 0: #competitor
                continue
            prob += self.site_landing(landing_product-1, np.zeros((5, 1), dtype=int), np.zeros((5, 1), dtype=int))
        return prob/self.n_sim


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
        bought_nodes[landing_product] = 1
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
    np.random.seed(42)
    n_sim = 10000
    env = RandomEnvironment(con_matrix, connectivity_matrix, prob_buy, n_sim)
    """ import matplotlib.pyplot as plt
    plt.plot(env.alpha_function(5, 100, 20)(np.linspace(-5, 200, 2000)))
    plt.show() """
    env2 = Environment(con_matrix, connectivity_matrix, prob_buy)
    probs = env.round().flatten()
    probs2 = env2.round()
    print('Random environment:', probs, 'Efficient environment: ', probs2, sep='\n')
