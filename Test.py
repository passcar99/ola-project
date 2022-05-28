from TS_Learner5D import *
from Environment import *
import math


Prob_Buy=np.array([1,1,1,1,1])
Con_matrix=np.array([[0,0.5,0.5,0,0],[0,0,0,0.5,0],[0,0,0,0,0.5],[0,0,0,0,0],[0,0,0,0,0]])
conpam_matrix=np.array([[1,.1,.1,.1,.1,.1]])#hhigher density on 1

env=Environment(conpam_matrix,Con_matrix,Prob_Buy,[1,1,1,2,0],[1,1,1,1,1])

arms_budgets=[0,1]

learner=TS_Learner5D(2)
max_reward=0;
vect_budget=[];

Result=[]
for k in range(32):#this should be a Clayrvoiant algorithm at all effects
    res = [int(i) for i in list('{0:0b}'.format(k))];
    if k==0:
        zero_to_add=5
    else:
        zero_to_add=math.ceil(5-math.log(k,2))
        
    for l in range(zero_to_add):
        res.insert(0,0)
    DirPar=conpam_matrix[0]+res
    ParSum=np.sum(DirPar)
    alpha_mean = [i/ParSum for i in DirPar]
    res[0]=env.round(alpha_mean)
    Result.append(res)

print(Result)
print("------------------------------------------")

for i in range(1000):
    multi_arm_idx=learner.pull_arm([])
    budget=[arms_budgets[multi_arm_idx[0]],arms_budgets[multi_arm_idx[1]],arms_budgets[multi_arm_idx[2]],arms_budgets[multi_arm_idx[3]],arms_budgets[multi_arm_idx[4]]]
    vect_budget.append(budget)
    reward=env.pull_arm(budget)
    learner.update(multi_arm_idx,reward)

   
print(vect_budget[-10:])

