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
        self.n_arms = len(self.arms)
        for product in range(self.n_products):
            self.expected_margins[product] = self.env.simplified_round(product, n_sim = 10000)


    # features: list of names like ["business", "region"]
    def compute_split(self, user_category_data, features, grouped_classes, fixed_groups = None):
        value_before_split = self.evaluate_split(user_category_data, grouped_classes)
        values_after_split = []
        partition=[]
        for feature in features:
            max = np.max(grouped_classes)
            partition.append([])
            for i, user_category in enumerate(user_category_data):
                partition[-1].append(max+1 if user_category.features[feature] and (fixed_groups is None or not fixed_groups[i] ) else grouped_classes[i])
            values_after_split.append(self.evaluate_split(user_category_data, partition[-1]))
        maximum = np.max(values_after_split)
        print(maximum, value_before_split)
        if maximum > value_before_split:
            argmax = np.argmax(values_after_split)
            left_user_category_data = [ cat_data.features[argmax]==0 and (fixed_groups is None or not fixed_groups[i]) for i, cat_data in enumerate(user_category_data)  ]
            """ classes_id_left = [c.class_id for i,  c in enumerate(user_category_data) if left_user_category_data[i] ]
            classes_id_right = [c.class_id for i,  c in enumerate(user_category_data) if not left_user_category_data[i]] """
            print("->",grouped_classes)
            grouped_classes[:] = np.array(partition[argmax])
            print("-->",grouped_classes)

            if len(features) > 1:
                new_feature_list = deepcopy(features)
                new_feature_list.remove(features[argmax])
                grouped_classes_copy = np.copy(grouped_classes)
                max_l = self.compute_split(user_category_data, new_feature_list, grouped_classes, np.logical_not(left_user_category_data) )#left
                new_feature_list = deepcopy(features)
                new_feature_list.remove(features[argmax]) 
                max_r = self.compute_split(user_category_data, new_feature_list, grouped_classes_copy, left_user_category_data)
                print(grouped_classes_copy, grouped_classes)
                max_all = self.compute_split(user_category_data, new_feature_list, grouped_classes, left_user_category_data)
                if max_r > maximum and max_r > max_l and max_r>max_all:
                    grouped_classes[:] = grouped_classes_copy
        return np.max([maximum, value_before_split])
        
            

    # user_category_data List[UserDataContext]
    def evaluate_split(self, user_category_data, class_partition):
        
        n_user_classes = len(user_category_data)
        n_splits = max(len(np.unique(class_partition)), 1)
        if len(class_partition)==0:
            class_partition = [0]*n_user_classes
        class_partition = np.array(class_partition) - np.min(class_partition)
        T = len(user_category_data[0].pulled_arms[0])
        print("--->", class_partition)
        pulled_arms = np.zeros((n_splits, self.n_products, T))
        rewards_per_product = np.zeros((n_splits, self.n_products, T))

        n_users = np.zeros((n_splits))
        # populate list of data
        for i, user_cat in enumerate(user_category_data):
            split = class_partition[i]
            n_users[split] += user_cat.n_users
            for product in range(self.n_products):
                pulled_arms[split][product]+=np.array(user_cat.pulled_arms[product]).flatten()
                rewards_per_product[split][product]+=np.array(user_cat.rewards_per_product[product])* user_cat.n_users
        value_matrix = np.zeros((n_splits*self.n_products, len(self.arms)))

        
        for i in range(n_splits): # for every feature value
            rewards_per_product[i] /= n_users[i]
            for product in range(self.n_products): # for every product
                x = np.around(pulled_arms[i][product]).flatten()
                unique_values_and_counts = np.unique(x, return_counts=True)
                T = len(pulled_arms[i][product])
                confidence = np.sqrt(-np.log(0.95)/unique_values_and_counts[1]/2) # confidence=0.95
                y = np.array(rewards_per_product[i][product])
                means = np.hstack([np.mean(y[x==v]) for v in unique_values_and_counts[0]])
                bounds =  means-confidence
                bounds[bounds<0]=0
                row = i*self.n_products+product
                if(len(unique_values_and_counts[0]) >1):
                    fun = interp1d(unique_values_and_counts[0], bounds, bounds_error=False, fill_value=(bounds[0], bounds[-1]))
                    lower_bounds = fun(self.arms)
                else:
                    lower_bounds = bounds # everything gets the same value
                
                value_matrix[row] = lower_bounds* self.expected_margins[product] * n_users[i]
                value_matrix[row, self.unfeasible_arms[product]] = -np.inf
        value_after_split = budget_allocations(value_matrix, self.arms, subtract_budget=True)[1]
        return value_after_split
        




        