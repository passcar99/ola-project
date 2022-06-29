import numbers
from TS_Learner5D import *
from GPTS_Learner import *
from Environment import *
import math
import matplotlib.pyplot as plt
import sys


Prob_Buy=np.array([1,0.5,1,1,1])
Con_matrix=np.array([[0,0.5,0.3,0,0],[0,0,0,0.5,0.3],[0.3,0,0,0,0.5],[0.5,0.3,0,0,0],[0.3,0.5,0,0,0]])
conpam_matrix = [
        {"alpha_params": [(0, 10, 20), (2, 15, 20),(2, 20, 20),(2, 15, 20),(1, 15, 20)], 
        "features":[0, 0], "total_mass":80, "avg_number":100}, 
                    ]
env=Environment(conpam_matrix,Con_matrix,Prob_Buy,[1,1,1,2,0],[1,1,1,1,1])
#FOR DEBUG PURPOSES THIS BINARY CASE CAN BE EASILY REPRESENTED IN 1D PASSING FROM BINARY TO INTEGER AND PLOTTING CUMULATIVE DATA AND
#EVENTUALLY THE BETA DISTRIBUTION
arms_budgets=np.array([0,10])#([[0,0.25,0.5,0.75,1],[0,0.25,0.5,0.75,1],[0,0.25,0.5,0.75,1],[0,0.25,0.5,0.75,1],[0,0.25,0.5,0.75,1]])
n_arms=len(arms_budgets)
n_super_arms=pow(n_arms,5)

Result=[]

for m in range(n_arms):
    for l in range(n_arms):
        for k in range(n_arms):
            for j in range(n_arms):
                for i in range(n_arms):
                    res=[0,arms_budgets[i],arms_budgets[j],arms_budgets[k],arms_budgets[l],arms_budgets[m]]
                    DirPar=conpam_matrix[0]+res
                    ParSum=np.sum(DirPar)
                    alpha_mean = [i/ParSum for i in DirPar]
                    res[0]=env.round(alpha_mean)
                    Result.append(res)

Clayrvoiant=np.zeros(n_super_arms);
for i in range(n_super_arms):
    Clayrvoiant[i]=Result[i][0]
print("------------------------------------------")
Cumulative=np.zeros(n_super_arms);
LastCum=np.zeros(n_super_arms);
Nn=300;
Regret=np.zeros(Nn+1);
# TS 
#learner=TS_Learner5D(n_arms)
#for i in range(Nn):
#    multi_arm_idx=learner.pull_arm([])
#    budget=[arms_budgets[0][multi_arm_idx[0]],arms_budgets[1][multi_arm_idx[1]],arms_budgets[2][multi_arm_idx[2]],arms_budgets[3][multi_arm_idx[3]],arms_budgets[4][multi_arm_idx[4]]]
    #budget.reverse()
    #Number=int("".join(str(i) for i in budget),2)
    #budget.reverse()
    
#    Number=multi_arm_idx[0]+n_arms*multi_arm_idx[1]+pow(n_arms,2)*multi_arm_idx[2]+pow(n_arms,3)*multi_arm_idx[3]+pow(n_arms,4)*multi_arm_idx[4]
#    Cumulative[Number]=Cumulative[Number]+1
#    reward=env.pull_arm(budget)
#    learner.update(multi_arm_idx,reward)
#    if i>(Nn-100):
#        LastCum[Number]=LastCum[Number]+1
        

#plt.plot(Cumulative/Nn, label="TS % pulled")

learner=GPTS_Learner5D(n_arms,arms_budgets/max(arms_budgets));

for i in range(Nn):
    sys.stdout.write('\r')
    multi_arm_idx=learner.pull_arm([])
    budget=[arms_budgets[multi_arm_idx[0]],arms_budgets[multi_arm_idx[1]],arms_budgets[multi_arm_idx[2]],arms_budgets[multi_arm_idx],arms_budgets[multi_arm_idx[4]]]
    reward=env.pull_arm(budget)
    Number=multi_arm_idx[0]+n_arms*multi_arm_idx[1]+pow(n_arms,2)*multi_arm_idx[2]+pow(n_arms,3)*multi_arm_idx[3]+pow(n_arms,4)*multi_arm_idx[4]
    #Regret[i+1]=Regret[i]+max(Clayrvoiant)-reward actual for the leatner mean for the clayrvoiant
    Regret[i+1]=Regret[i]+max(Clayrvoiant)-Clayrvoiant[Number] #mean for both
    learner.update(multi_arm_idx,reward/max(Clayrvoiant))#REWARD NORMALIZATION

    sys.stdout.write("{:.1f}%".format((100/(Nn-1)*i)))
    sys.stdout.flush()

CumFast=np.array([ len(listElem) for listElem in learner.rewards_per_arm])
plt.figure(1)
plt.plot(Clayrvoiant, label="Expected Value")
plt.plot(CumFast/Nn, label="GPTS % pulled FAST")
plt.legend(loc="upper left")
plt.show()

plt.figure(2)
plt.plot(Regret, label="Regret")
plt.show()
