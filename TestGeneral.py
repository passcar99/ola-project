import numbers
from TS_Learner5D import *
from GPTS_Learner import *
from Environment import *
import math
import matplotlib.pyplot as plt
import sys


Prob_Buy=np.array([1,0.5,1,1,1])
Con_matrix=np.array([[0,0.5,0.3,0,0],[0,0,0,0.5,0.3],[0.3,0,0,0,0.5],[0.5,0.3,0,0,0],[0.3,0.5,0,0,0]])
conpam_matrix=np.array([[1,.1,.1,.1,.1,.1]])#hhigher density on 1

env=Environment(conpam_matrix,Con_matrix,Prob_Buy,[1,1,1,2,0],[1,1,1,1,1])
#FOR DEBUG PURPOSES THIS BINARY CASE CAN BE EASILY REPRESENTED IN 1D PASSING FROM BINARY TO INTEGER AND PLOTTING CUMULATIVE DATA AND
#EVENTUALLY THE BETA DISTRIBUTION
arms_budgets=np.array([[0,0.5,1],[0,0.5,1],[0,0.5,1],[0,0.5,1],[0,0.5,1]])
n_arms=len(arms_budgets[0])
n_super_arms=pow(n_arms,5)

Result=[]

for m in range(n_arms):
    for l in range(n_arms):
        for k in range(n_arms):
            for j in range(n_arms):
                for i in range(n_arms):
                    res=[0,arms_budgets[0][i],arms_budgets[1][j],arms_budgets[2][k],arms_budgets[3][l],arms_budgets[4][m]]
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
Nn=365;
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

learner=GPTS_Learner(n_arms,arms_budgets);
Cumulative=np.zeros(n_super_arms);
LastCum=np.zeros(n_super_arms);


for i in range(Nn):
    sys.stdout.write('\r')
    multi_arm_idx=learner.pull_arm([])
    budget=[arms_budgets[0][multi_arm_idx[0]],arms_budgets[1][multi_arm_idx[1]],arms_budgets[2][multi_arm_idx[2]],arms_budgets[3][multi_arm_idx[3]],arms_budgets[4][multi_arm_idx[4]]]
    #budget.reverse()
    #Number=int("".join(str(i) for i in budget),2)
    #budget.reverse()
    Number=multi_arm_idx[0]+n_arms*multi_arm_idx[1]+pow(n_arms,2)*multi_arm_idx[2]+pow(n_arms,3)*multi_arm_idx[3]+pow(n_arms,4)*multi_arm_idx[4]

    Cumulative[Number]=Cumulative[Number]+1
    reward=env.pull_arm(budget)
    Regret[i+1]=Regret[i]+max(Clayrvoiant)-reward
    learner.update(multi_arm_idx,reward)
    if i>(Nn-100):
        LastCum[Number]=LastCum[Number]+1
    sys.stdout.write("{:.1f}%".format((100/(Nn-1)*i)))
    sys.stdout.flush()


plt.figure(1)
plt.plot(Clayrvoiant, label="Expected Value")
plt.plot(Cumulative/Nn, label="GPTS % pulled")
plt.legend(loc="upper left")
plt.show()
plt.figure(2)
plt.plot(Regret, label="Regret")
plt.show()
