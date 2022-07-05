import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
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
        :param bounds: lower and upper bounds for each product (n_products*2 matrix).
        :param environment_type: type of environment to use to estimate the expected margin.
        """
        self.user_classes = []
        for user_class in conpam_matrix:
            self.user_classes.append(UserCategory(**user_class))
        self.con_matrix=con_matrix
        self.lam=0.5;#implicit in Con_matrix
        self.prob_buy=np.array(prob_buy)
        self.avg_sold = np.array(avg_sold)
        self.margins = np.array(margins)
        self.n_prods = len(self.margins)
        self.recursion = 0
        self.t = 0
        

    def round(self, budgets):
        """ 
        Function returning the feedback from a round in the basic case.
        :param budgets: superarm (list of budgets for each product).
        :return: list of dictionaries. Each element of the list contains the number of users,
         the alphas realization and the profit for the corresponding user category.
        """
        category_realizations = [{} for _ in range(len(self.user_classes))]

        for i, user_class in enumerate(self.user_classes):
            n_users = np.random.poisson(user_class.avg_number)
            betas = user_class.get_alpha_from_budgets(budgets, self.t)
            non_zero_prods = np.nonzero(betas)
            alphas = stats.dirichlet.rvs(betas[non_zero_prods], size=1)[0]
            alphas_tilde = np.zeros((self.n_prods+1))
            alphas_tilde[non_zero_prods] = alphas
            category_realizations[i]['n_users'] = n_users
            category_realizations[i]['alphas'] = alphas_tilde # prolong Dirichlet to values of alpha_i ==0

        for n in category_realizations:
            cusum = np.zeros((5))
            for _ in tqdm(range(n['n_users']), leave=False):
                landing_product = np.nonzero(np.random.multinomial(1, n['alphas']))[0][0]
                self.recursion = 1
                if landing_product == 0: #competitor
                    continue
                activated_nodes = np.zeros((5), dtype=int)
                bought_nodes = np.zeros((5))
                cusum += self.site_landing(landing_product-1, activated_nodes , bought_nodes)
            n['profit'] = cusum.flatten().dot(self.margins.transpose()) - np.sum(budgets)
        self.t += 1
        return category_realizations

    def round_step_4(self, budgets):
        """ 
        Function returning the feedback from a round in the step_4 case.
        :param budgets: superarm (list of budgets for each product).
        :return: list of dictionaries. Each element of the list contains the number of users,
         the alphas realization, the number of items sold to each user and the profit for the corresponding user category.
        """
        category_realizations = [{} for _ in range(len(self.user_classes))]
        for i, user_class in enumerate(self.user_classes):
            n_users = np.random.poisson(user_class.avg_number)
            betas = user_class.get_alpha_from_budgets(budgets, self.t)
            non_zero_prods = np.nonzero(betas)
            alphas = stats.dirichlet.rvs(betas[non_zero_prods], size=1)[0]
            alphas_tilde = np.zeros((self.n_prods+1))
            alphas_tilde[non_zero_prods] = alphas
            category_realizations[i]['n_users'] = n_users
            category_realizations[i]['alphas'] = alphas_tilde # prolong Dirichlet to values of alpha_i ==0

        for user_category in category_realizations:
            cusum = np.zeros((5))
            items_sold = np.zeros((user_category['n_users'], 5))
            for _ in tqdm(range(user_category['n_users']), leave=False):
                landing_product = np.nonzero(np.random.multinomial(1, user_category['alphas']))[0][0]
                if landing_product == 0: #competitor
                    continue
                self.recursion = 1
                activated_nodes = np.zeros((5), dtype=int)
                bought_nodes = np.zeros((5))
                cusum += self.site_landing(landing_product-1, activated_nodes , bought_nodes)
                items_sold[i, :] = bought_nodes
            user_category['items'] = items_sold
            user_category['profit'] = cusum.flatten().dot(self.margins.transpose()) - np.sum(budgets)
        self.t += 1
        return category_realizations

    def round_step_5(self, budgets):
        """ 
        Function returning the feedback from a round in the step_5 case.
        :param budgets: superarm (list of budgets for each product).
        :return: list of dictionaries. Each element of the list contains the number of users,
         the alphas realization, the number of items sold to each user, the history of activated nodes for each visit
          and the profit for the corresponding user category.
        """
        category_realizations = [{} for _ in range(len(self.user_classes))]
        
        for i, user_class in enumerate(self.user_classes):
            n_users = np.random.poisson(user_class.avg_number)
            betas = user_class.get_alpha_from_budgets(budgets, self.t)
            non_zero_prods = np.nonzero(betas)
            alphas = stats.dirichlet.rvs(betas[non_zero_prods], size=1)[0]
            alphas_tilde = np.zeros((self.n_prods+1))
            alphas_tilde[non_zero_prods] = alphas
            category_realizations[i]['n_users'] = n_users
            category_realizations[i]['alphas'] = alphas_tilde # prolong Dirichlet to values of alpha_i ==0

        for user_category in category_realizations:
            cusum = np.zeros((5))
            activation_history = np.zeros((user_category['n_users'], 5))
            for i in tqdm(range(user_category['n_users']), leave=False):
                landing_product = np.nonzero(np.random.multinomial(1, user_category['alphas']))[0][0]
                if landing_product == 0: #competitor
                    continue
                self.recursion = 1
                activated_nodes = np.zeros((5), dtype=int)
                bought_nodes = np.zeros((5))
                cusum += self.site_landing(landing_product-1, activated_nodes , bought_nodes)
                activation_history[i, :] = activated_nodes
            user_category['profit'] = cusum.flatten().dot(self.margins.transpose()) - np.sum(budgets)
            user_category['activation_history'] = activation_history
        self.t += 1
        return category_realizations

    def round_step_7(self, budgets):
        """ 
        Function returning the feedback from a round in the step_7 case.
        :param budgets: superarm (list of budgets for each product).
        :return: list of dictionaries. Each element of the list contains the number of users,
         the alphas realization, the number of items sold to each user, the feature values and the profit for the corresponding user category.
        """
        category_realizations = [{} for _ in range(len(self.user_classes))]
        
        for i, user_class in enumerate(self.user_classes):
            n_users = np.random.poisson(user_class.avg_number)
            betas = user_class.get_alpha_from_budgets(budgets, self.t)
            non_zero_prods = np.nonzero(betas)
            alphas = stats.dirichlet.rvs(betas[non_zero_prods], size=1)[0]
            alphas_tilde = np.zeros((self.n_prods+1))
            alphas_tilde[non_zero_prods] = alphas
            category_realizations[i]['n_users'] = n_users
            category_realizations[i]['alphas'] = alphas_tilde # prolong Dirichlet to values of alpha_i ==0

        for cat_idx , user_category in enumerate(category_realizations):
            cusum = np.zeros((5))
            items_sold = np.zeros((user_category['n_users'], 5))
            for i in tqdm(range(user_category['n_users']), leave=False):
                landing_product = np.nonzero(np.random.multinomial(1, user_category['alphas']))[0][0]
                if landing_product == 0: #competitor
                    continue
                self.recursion = 1
                activated_nodes = np.zeros((5), dtype=int)
                bought_nodes = np.zeros((5))
                cusum += self.site_landing(landing_product-1, activated_nodes , bought_nodes)
                items_sold[i, :] = bought_nodes
            user_category['items'] = items_sold
            user_category['profit'] = cusum.flatten().dot(self.margins.transpose()) - np.sum(budgets)
            user_category['features'] = self.user_classes[cat_idx].features
        self.t +=1
        return category_realizations
    



    def alpha_functions(self):
        alpha_functions = []
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
        while num_bought==0:
            num_bought = np.random.poisson(self.avg_sold[landing_product])
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


