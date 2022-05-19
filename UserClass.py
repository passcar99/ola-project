from typing import Tuple, List
from scipy.interpolate import interp1d
import numpy as np

class UserClass():

    """ 
    Class representing a category of users. 
    To initialize pass a list of size "number of alphas - 1", a list of features 
    and the total mass for the Dirichlet distribution (to control variance) TODO remove if not necessary, check
    ES: alpha_params = [(0, 10, 0.1), (5, 20, 0.5), ...]
    """
    def __init__(self, alpha_params: List[Tuple], features: List, total_mass) -> None:
        self.alpha_functions = []
        for tuple in alpha_params:
            self.alpha_functions.append(self.alpha_function(*tuple))
        self.features = features
        self.total_mass = total_mass
        

    def alpha_function(self, min_budget, max_budget, alpha_bar): #assuming linear behaviour. TODO check
        x1, y1 = min_budget, 0
        x2, y2 = max_budget, alpha_bar
        #return a function to be called as f(inputs) where inputs can be a number or an array
        return interp1d([x1, x2], [y1, y2], kind='linear', bounds_error=False, fill_value=(y1, y2))


    def get_alpha_from_budgets(self, budgets: List):
        alphas = []
        for i, bdg in enumerate(budgets):
            alphas.append(self.alpha_functions[i](bdg))
        alphas.insert(0, self.total_mass-sum(alphas)) #alpha0
        return np.array(alphas, dtype=np.double)

""" Just for testing. """
if __name__=='__main__':
    user_class = UserClass([(0, 10, 2), (5, 10, 6)], [0, 0], 10)
    print(user_class.get_alpha_from_budgets([20, 7.5]))