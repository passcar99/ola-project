import numpy as np
from scipy import stats
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
        alphas=stats.dirichlet.rvs(self.conpam_matrix[0], size=1, random_state=42)
        alphas[0];#to competitors
        prob = np.zeros((5, 1))
        for i in tqdm(range(self.n_sim)):
            landing_product = np.nonzero(np.random.multinomial(1, alphas[0][1:])) # do not consider competitor, take only nonzero index
            prob += self.site_landing(landing_product, np.zeros((5, 1)), np.zeros((5, 1)))
        return prob/self.n_sim



    def site_landing(self,landing_product,activated_nodes, bought_nodes):#this case is with only ONE quantity bought and all the item have same price
        activated_nodes[landing_product] = 1
        buy = np.random.binomial(1, self.prob_buy[landing_product])
        if buy == 0:
            return bought_nodes
        bought_nodes[landing_product] = 1
        sec_prod_prob = self.con_matrix[landing_product].flatten()
        secondary_products = np.argsort(sec_prod_prob)
        first_sec = secondary_products[-1]
        second_sec = secondary_products[-2]
        move = np.random.binomial(1, sec_prod_prob[first_sec]*(1-activated_nodes[first_sec])) # if 0 should return always 0, TODO check
        if move:
            self.site_landing(first_sec, activated_nodes, bought_nodes)
        move = np.random.binomial(1, sec_prod_prob[second_sec]*(1-activated_nodes[second_sec])) # if 0 should return always 0, TODO check
        if move:
            self.site_landing(second_sec, activated_nodes, bought_nodes)
        return bought_nodes

"""Just for testing""" 
if __name__=='__main__':
    con_matrix = [[0.2, 0.10, 0.15, 0.2, 0.22, 0.13]]
    connectivity_matrix = np.array([[0, 0.2, 0.4, 0.3, 0.1],
                                    [0.5, 0, 0.1, 0.3, 0.1],
                                    [0.3, 0.2, 0, 0.1, 0.4],
                                    [0.13, 0.17, 0.30, 0, 0.4],
                                    [0.16, 0.34, 0.15, 0.25, 0],
                                    ])
    prob_buy = np.array([0.1, 0.2, 0.5, 0.9, 0.7])
    n_sim = 10000
    env = RandomEnvironment(con_matrix, connectivity_matrix, prob_buy, n_sim)
    env2 = Environment(con_matrix, connectivity_matrix, prob_buy)
    probs = env.round()
    probs2 = env2.round()
    print(probs, probs2)
