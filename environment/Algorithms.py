import numpy as np


def budget_allocations(value_matrix, budgets, subtract_budget=False):
    """ 
    Dynamic programming algorithm for determining the optimal allocation of budgets to the different subcampaigns.
    It exploits a tabular representation of the solution.
    :param value_matrix: |products|x|budgets| matrix containting for each product i its value if budget j is spent on it
    -inf = -np.inf
    :param budgets: list of budgets,
    :param subtract_budgets: whether to subtract the budget from the last row and consider also the advertising costs or not.

    :return : best combination of budgets and (expected) value for it.
    """
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
            temp_sol = []
            for j, b_j in enumerate(budgets[:i+1]):
                if b_j+budgets[i-j] <= b_i: # accomodate for when the budgets do not sum to the current one TODO check
                    values.append(solution_value[product, j]+value_matrix[product, i-j]) #solution values starts from product 0
                    temp_sol.append(solution[j].copy())
                    temp_sol[-1][product] = b_i-b_j
                #else:
                    #values.append(solution_value[product-1, j]+value_matrix[product, i-j]) #if previous one is not a feasible solution according to budget
            
            best_comb = np.argmax(values) # TODO break ties
            solution_value[product+1, i] = values[best_comb]

            """ sol = np.zeros(shape=(n_products, 1))
            sol = solution[best_comb]
            sol[product] = budgets[i -best_comb] """
            sol = temp_sol[best_comb]
            new_solution.append(sol)
        solution = new_solution
    if subtract_budget:
        solution_value[n_products] -= np.array(budgets)
    best_comb = np.argmax(solution_value[n_products])
    return solution[best_comb], solution_value[n_products, best_comb]

def clairvoyant(environment, arms, bounds, total_mass=100):
    """ 
    Clairvoyant algorithm. Given an environment, a set of arms, per product bounds, get the best allocation of budgets to campaigns.
    """
    n_arms = len(arms)
    n_products = environment.n_prods
    value_matrix = np.zeros((n_products, n_arms))
    unfeasible_arms = []
    for p in range(n_products):
        unfeasible_arms.append(np.logical_or(arms <= bounds[p][0], arms >= bounds[p][1]))
    alpha_functions = np.array([ fun(arms) for fun in environment.alpha_functions()[0]])
    alpha_functions = alpha_functions/environment.user_classes[0].total_mass #total_mass
    expected_margin = np.zeros((n_products))
    for p in range(n_products):
        expected_margin[p] = environment.simplified_round(p, n_sim = 100000)
        value_matrix[p, :] = alpha_functions[p, :]* expected_margin[p] *100 # n_users
        value_matrix[p, unfeasible_arms[p]] = -np.inf
    return budget_allocations(value_matrix, arms, subtract_budget=True)

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


                    

