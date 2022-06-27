import numpy as np
from copy import deepcopy
""" 
value_matrix: |products|x|budgets| matrix containting for each product i its value if budget j is spent on it
-inf = -np.inf
"""
def budget_allocations(value_matrix, budgets, subtract_budget=False):
    n_products = value_matrix.shape[0]
    n_budgets = value_matrix.shape[1]
    solution_value = np.zeros(shape=(n_products+1, n_budgets)) #actually we need just the previous row (TODO make more efficient later)
    solution = [] #list of solutions as [[[0, 20], [1, 40]], ...]
    best_comb = 0
    solution_value[1, :] = value_matrix[0, :] # initialize second row with first product
    for budg in budgets:
        sol = np.zeros(shape=(n_products, 1))
        sol[0] = budg # unfeasible solutions should be ruled out by argmax and -np.inf TODO check
        solution.append(sol)
    for product in range(1, n_products):
        new_solution = []
        for i, b_i in enumerate(budgets):
            values = []
            for j, b_j in enumerate(budgets[:i+1]):
                if b_j+budgets[i-j] <= b_i: # accomodate for when the budgets do not sum to the current one TODO check
                    values.append(solution_value[product, j]+value_matrix[product, i-j]) #solution values starts from product 0
                else:
                    values.append(solution_value[product-1, j]+value_matrix[product, i-j]) #if previous one is not a feasible solution according to budget
            best_comb = np.argmax(values) # TODO break ties
            sol = np.zeros(shape=(n_products, 1))
            sol = solution[best_comb]
            sol[product] = budgets[i -best_comb]
            new_solution.append(sol)
            solution_value[product+1, i] = np.max(values)
        solution = new_solution
    if subtract_budget:
        solution_value[n_products] -= np.array(budgets)
    best_comb = np.argmax(solution_value[n_products])
    return solution[best_comb], solution_value[n_products, best_comb]



if __name__=='__main__':
    value_matrix = np.array([[-np.inf, 90, 100, 105, 110, -np.inf, -np.inf, -np.inf],
                            [ 0, 82, 90, 92, -np.inf, -np.inf, -np.inf, -np.inf],
                            [0, 80, 83, 85, 86, -np.inf, -np.inf, -np.inf],
                            [-np.inf, 90, 110, 115, 118, 120, -np.inf, -np.inf],
                            [-np.inf, 111, 130, 138, 142, 148, 155, -np.inf]
                            ])
    budgets = [0, 10, 20, 30, 40, 50, 60, 70]
    print(budget_allocations(value_matrix, budgets))
    print('Subtract budget')
    print(budget_allocations(value_matrix, budgets, True))


                    

