import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from environment.Algorithms import budget_allocations
from environment.Environment import Environment
from copy import deepcopy
from scipy.interpolate import interp1d

class ContextGeneration():

    def __init__(self, n_products, arms, margins, environment, unfeasible_arms):
        self.n_products = n_products
        self.arms = arms
        self.margins = margins
        self.env = environment
        self.expected_margins = np.zeros((self.n_products))
        self.unfeasible_arms = unfeasible_arms
        for product in range(self.n_products):
            self.expected_margins[product] = self.env.simplified_round(product, n_sim = 10000)


    # features: list of names like ["business", "region"]
    def compute_split(self, user_category_data, features, grouped_classes):
        value_before_split = self.evaluate_split(user_category_data, -1)
        values_after_split = []
        for feature in features:
            values_after_split.append(self.evaluate_split(user_category_data, feature))
        maximum = np.max(values_after_split)
        print(maximum, value_before_split)
        if maximum > value_before_split:
            argmax = np.argmax(values_after_split)
            left_user_category_data = [ cat_data for cat_data in user_category_data if cat_data.features[argmax]==0]
            right_user_category_data = [ cat_data for cat_data in user_category_data if cat_data.features[argmax]==1]
            classes_id_left = [c.class_id for c in left_user_category_data]
            classes_id_right = [c.class_id for c in right_user_category_data]
            grouped_classes[classes_id_left] = np.max(grouped_classes)+1
            grouped_classes[classes_id_right]= np.max(grouped_classes)+1
            if len(features) > 1:
                new_feature_list = deepcopy(features)
                new_feature_list.remove(features[argmax])
                self.compute_split(left_user_category_data, new_feature_list, grouped_classes)
                new_feature_list = deepcopy(features)
                new_feature_list.remove(features[argmax]) 
                self.compute_split(right_user_category_data, new_feature_list, grouped_classes)
        
            

    # user_category_data List[UserDataContext]
    def evaluate_split(self, user_category_data, feature = -1):
        if feature == -1: # no split
            n_feature_values = 1
        else:
            n_feature_values = 2
        T = len(user_category_data[0].pulled_arms[0])
        pulled_arms = np.zeros((n_feature_values, self.n_products, T))
        rewards_per_product = np.zeros((n_feature_values, self.n_products, T))

        n_users = np.zeros((n_feature_values))
        # populate list of data
        for user_cat in user_category_data:
            feature_value = int(user_cat.features[feature]) if feature != -1 else 0
            n_users[feature_value] += user_cat.n_users
            for product in range(self.n_products):
                pulled_arms[feature_value][product]+=np.array(user_cat.pulled_arms[product]).flatten()
                rewards_per_product[feature_value][product]+=np.array(user_cat.rewards_per_product[product])* user_cat.n_users
        value_matrix = np.zeros((n_feature_values*self.n_products, len(self.arms)))
        for i in range(n_feature_values): # for every feature value
            rewards_per_product[i] /= n_users[i]
            for product in range(self.n_products): # for every product
                x = np.array(pulled_arms[i][product]).flatten()
                unique_values_and_counts = np.unique(x, return_counts=True)
                T = len(pulled_arms[i][product])
                confidence = np.sqrt(-np.log(0.95)/unique_values_and_counts[1]/2) # confidence=0.95
                print(unique_values_and_counts)
                y = np.array(rewards_per_product[i][product])
                means = np.hstack([np.mean(y[x==v]) for v in unique_values_and_counts[0].tolist()])
                bounds = means-confidence
                row = i*self.n_products+product
                if(len(unique_values_and_counts[0]) >1):
                    fun = interp1d(unique_values_and_counts[0], means-confidence, bounds_error=False, fill_value=(bounds[0], bounds[-1]))
                    lower_bounds = fun(self.arms)
                else:
                    lower_bounds = bounds # everything gets the same value
                
                value_matrix[row] = lower_bounds* self.expected_margins[product] * n_users[i]
                value_matrix[row, self.unfeasible_arms[product]] = -np.inf

        value_after_split = budget_allocations(value_matrix, self.arms, subtract_budget=True)[1]

        return value_after_split
        




        