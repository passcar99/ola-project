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
            best_comb = np.argmax(values) # TODO break ties
            solution_value[product+1, i] = values[best_comb]
            sol = temp_sol[best_comb]
            new_solution.append(sol)
        solution = new_solution
    if subtract_budget:
        solution_value[n_products] -= np.array(budgets)
    best_comb = np.argmax(solution_value[n_products])
    return solution[best_comb], solution_value[n_products, best_comb]


def regret_bound(Lambda, variance, horizon, n_campaigns=5, size_of_superarm=5, arm_dimension=1,delta=0.95, type="TS"):
    #gamma= O((log(t))^(d+1))
    time_instants = np.arange(1, horizon+1)
    B = 8*np.log(2*(time_instants*time_instants)*size_of_superarm*n_campaigns/delta)
    c = 2*Lambda**2/(np.log(1+1/variance))
    gamma_sum = n_campaigns*np.power(np.log(time_instants*size_of_superarm), arm_dimension+1)
    regret = np.sqrt(c*n_campaigns*time_instants*B*gamma_sum)
    return regret

def clairvoyant(environment, arms, bounds, total_mass=100, phase=None, class_mask=[]):
    """ 
    Clairvoyant algorithm. Given an environment, a set of arms, per product bounds, get the best allocation of budgets to campaigns.
    """
    n_arms = len(arms)
    n_products = environment.n_prods
    n_user_classes = len(environment.user_classes)
    value_matrix = np.zeros((n_products * (len(np.unique(class_mask)) if len(np.unique(class_mask))>0 else 1), n_arms))
    unfeasible_arms = []
    class_mask = np.array(class_mask)
    expected_margin = np.zeros((n_products))
    for p in range(n_products):
        unfeasible_arms.append(np.logical_or(arms <= bounds[p][0], arms >= bounds[p][1]))
        expected_margin[p] = environment.simplified_round(p, n_sim = 100000)
    if len(class_mask)==0:
        class_mask = [0]*n_user_classes
    for split in range(n_user_classes):
        user_class = environment.user_classes[split]
        factor = 1
        tot_users = np.sum([u_class.avg_number for i, u_class in enumerate(environment.user_classes) if class_mask[i] == class_mask[split]])
        factor = user_class.avg_number/tot_users
        alpha_functions = np.array([ fun(arms * factor) for fun in environment.alpha_functions(phase)[split]])
        alpha_functions = alpha_functions/user_class.total_mass #total_mass
        for p in range(n_products):
            row = (class_mask[split] if len(class_mask)>0 else 0)*n_products+p
            value_matrix[row, :] += alpha_functions[p, :]* expected_margin[p] *environment.user_classes[split].avg_number
            value_matrix[row, unfeasible_arms[p]] = -np.inf
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


                    

