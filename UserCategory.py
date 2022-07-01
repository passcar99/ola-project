from typing import Tuple, List
from scipy.interpolate import interp1d
import numpy as np

class UserCategory():

    """ 
    Class representing a category of users. 
    To initialize pass a list of size "number of alphas - 1", a list of features,
    the total mass for the Dirichlet distribution (to control variance) [TODO check]
    and avg_number the average number of users for that class [TODO change]
    ES: alpha_params = [(0, 10, 0.1), (5, 20, 0.5), ...]
    :param alpha_params: list of parameters for the alpha functions of each product.
    :param features: list of features.
    :param avg_number: average number of users for the current class.
    """
    def __init__(self, alpha_params: List[Tuple], features: List, total_mass, avg_number) -> None:
        self.alpha_functions = []
        for tuple in alpha_params:
            self.alpha_functions.append(self.alpha_function(*tuple))
        self.features = features
        self.total_mass = total_mass
        self.avg_number = avg_number
        


    def alpha_function(self, min_budget, max_budget, alpha_bar): #assuming linear behaviour. TODO check
        """ 
        Method returning tha alpha function for a given product and class as a piecewise linear function.
        """
        x1, y1 = min_budget, 0
        x2, y2 = max_budget, alpha_bar
        #return a function to be called as f(inputs) where inputs can be a number or an array
        return interp1d([x1, x2], [y1, y2], kind='linear', bounds_error=False, fill_value=(y1, y2))


    def get_alpha_from_budgets(self, budgets: List):
        """ 
        Method for computing the alphas for class given a superarm.
        """
        alphas = []
        for i, bdg in enumerate(budgets):
            alphas.append(self.alpha_functions[i](bdg))
        alphas.insert(0, self.total_mass-sum(alphas)) #alpha0
        return np.array(alphas, dtype=np.double).flatten()

""" Just for testing. """
if __name__=='__main__':
    user_class = UserCategory([(0, 10, 2), (5, 10, 6)], [0, 0], 10, 10)
    import matplotlib.pyplot as plt
    plt.plot(np.linspace(-5, 200, 2000), user_class.alpha_function(5, 100, 20)(np.linspace(-5, 200, 2000)))
    plt.show() 
    print(user_class.get_alpha_from_budgets([20, 7.5]))