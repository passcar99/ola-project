import numpy as np
from scipy import stats
from tqdm import tqdm
from .Environment import Environment
from .UserCategory import UserCategory
from typing import List, Dict

class RandomEnvironment():
    """ 
    Environment for simulating the behaviour of the different user classes at each round.

    """

    def __init__(self,conpam_matrix:List[Dict],con_matrix, prob_buy, avg_sold, margins):
        """ 
        :param arms: list of arms (budgets).
        :param conpam_matrix: a list describing each user class in terms of alpha function parameters, features, 
        average number of users, total mass of the Dirichlet parameters like [{"alpha_params": [(0, 10, 2), ...], "features":[0, 0], "total_mass":64, "avg_number":100},...]
        :param con_matrix: connectivity matrix of the graph.
        :param prob_buy: probability that an item is bought when displayed as primary.
        :param avg_sold: average quantity of items bought for each product when displayed as primary.
        :param margins: margin (profit) for each arm.
        """
        self.user_classes = []
        for user_class in conpam_matrix:
            self.user_classes.append(UserCategory(**user_class))
        self.con_matrix=con_matrix
        self.prob_buy=np.array(prob_buy)
        self.avg_sold = np.array(avg_sold)
        self.margins = np.array(margins)
        self.n_prods = len(self.margins)
        self.recursion = 0
        self.t = 0
        

        
    def round(self, budgets_per_class, observed_features=False):
        """ 
        Function returning the feedback from a round. It returns the data that may be required by learners for each step.
        :param budgets_per_class: in cases different than step 7 will be a superarm, in case of step 7 it will be an array of tuple ([user_classes],superarm) elements
        :param observed_features: whether or not user features are observed. True only for step 7.
        :return: list of dictionaries. Each element of the list contains the number of users,
         the alphas realization, the number of items sold to each user, activation status of each node for every visit,
         the feature values and the profit for the corresponding user category.
        """
        #To keep old code functioning---
        if observed_features==False:
            #check
            if len(budgets_per_class)!=5:
                raise Exception("budgets_per_class is not the common superarm")
            budgets_per_class=([[list(range(0,len(self.user_classes))),budgets_per_class]])
        #-------------------------------
        category_realizations = [{} for _ in range(len(self.user_classes))]
        n_users = np.zeros((len(self.user_classes)), np.int32)
        for i, user_class in enumerate(self.user_classes):
            n_users[i] = np.random.poisson(user_class.avg_number)
            category_realizations[i]['n_users'] = n_users[i]
        for cat_idx , user_category in enumerate(category_realizations):
            budgets=[-1]
            for cases in budgets_per_class:
                if cat_idx in cases[0]:#Found the user_class into budgets_per_class
                    budgets=np.array(cases[1])*n_users[cat_idx]/n_users[cases[0]].sum()
            if budgets[0]==-1:
                raise Exception("A user is missing from budgets_per_class")
            items_sold = np.zeros((user_category['n_users'], 5))+np.nan
            activation_history = np.zeros((user_category['n_users'], 5))
            betas = user_class.get_alpha_from_budgets(budgets, self.t)
            betas[betas<0]=0
            non_zero_prods = np.nonzero(betas)
            alphas = stats.dirichlet.rvs(betas[non_zero_prods], size=1)[0]
            alphas_tilde = np.zeros((self.n_prods+1))
            alphas_tilde[non_zero_prods] = alphas
            user_category['alphas'] = alphas_tilde
            for i in tqdm(range(user_category['n_users']), leave=False):
                landing_product = np.nonzero(np.random.multinomial(1, user_category['alphas']))[0][0]
                if landing_product == 0: #competitor
                    continue
                self.recursion = 1
                activated_nodes = np.zeros((5), dtype=int)
                bought_nodes = np.ones((5))*(np.nan)
                self.site_landing(landing_product-1, activated_nodes , bought_nodes)
                items_sold[i, :] = bought_nodes
                activation_history[i, :] = activated_nodes
            user_category['items'] = items_sold
            bought_quantities = np.nansum(items_sold, axis=0)
            user_category['profit'] = bought_quantities.flatten().dot(self.margins.transpose()) - np.sum(budgets)
            user_category['activation_history'] = activation_history
            if observed_features:
                user_category['features'] = self.user_classes[cat_idx].features
        self.t +=1
        return category_realizations
    



    def alpha_functions(self, phase=None):
        alpha_functions = []
        if phase is not None: 
            for user_cat in self.user_classes:
                alpha_functions.append(user_cat.alpha_functions[phase])
        else:
            for user_cat in self.user_classes:
                alpha_functions.append(user_cat.alpha_functions)
        return alpha_functions

    def site_landing(self,landing_product,activated_nodes, bought_nodes):
        """ 
        Method inplementing the visit of the website buy a user.
        :param landing_product: current product displayed as primary.
        :param activated_nodes: array of already visited nodes. 0 if not visited, otherwise their level in the recursion.
        :param bought_nodes: number of items bought for each product during the current visit.
        """
        rec_level = self.recursion
        activated_nodes[landing_product] = rec_level
        buy = np.random.binomial(1, self.prob_buy[landing_product])
        if buy == 0:
            return bought_nodes
        num_bought = np.random.poisson(self.avg_sold[landing_product])
        """ while num_bought==0:
            num_bought = np.random.poisson(self.avg_sold[landing_product]) """
        bought_nodes[landing_product] = num_bought
        sec_prod_prob = self.con_matrix[landing_product].flatten()
        secondary_products = np.argsort(sec_prod_prob)
        if(len(secondary_products)==0) :
            return bought_nodes
        first_sec = secondary_products[-1]
        second_sec = secondary_products[-2]
        move = np.random.binomial(1, (1-(activated_nodes[first_sec]!=0))*sec_prod_prob[first_sec]) # if 0 returns always 0
        if move:
            self.recursion+=1
            self.site_landing(first_sec, activated_nodes, bought_nodes)
            self.recursion = rec_level
        move = np.random.binomial(1, (1-(activated_nodes[second_sec] !=0))*sec_prod_prob[second_sec]) # if 0 returns always 0
        if move:
            self.recursion+=1
            self.site_landing(second_sec, activated_nodes, bought_nodes)
            self.recursion = rec_level
        return bought_nodes


    def simplified_round(self, product, n_sim):
        """ 
        Function for computing the expected margin of a product. 
        :param product: product whose margin must be computed.
        :param n_sim: number of Monte Carlo simulations.
        """
        cusum = np.zeros(len(self.margins))
        for _ in range(n_sim):
            activated_nodes = np.zeros((5), dtype=int)
            bought_nodes = np.zeros((5))
            self.recursion = 1
            cusum += self.site_landing(product, activated_nodes, bought_nodes)
        expected_margin = cusum.flatten().dot(self.margins.transpose())/n_sim
        return expected_margin



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
    env2 = Environment(con_matrix, connectivity_matrix, prob_buy, [10]+avg_sold, [10]+margins)
    probs = env.round_step_7([10, 20, 6,50,45])
    #probs2 = env2.round()
    print('Random environment:', probs, )#'Efficient environment: ', probs2, sep='\n')
    print(env2.pull_arm([10, 20, 6,50,45]))


