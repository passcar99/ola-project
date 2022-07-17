import json
import matplotlib.pyplot as plt
import os
import numpy as np
path="../STEP6_1/backup/Step6"

opt=[ 5408.51196,     8849.84628,    72176.05288485]

with open(path+"/GPUCB_Learner_detecting_ex-1") as fp:
        ex_reward=json.load(fp)
with open(path+"/GPUCB_Learner_detecting-1") as fp:
        reward=json.load(fp)

plt.figure()
plt.plot(ex_reward[0])
plt.plot(reward[0])
plt.legend(["exact reward","reward"])
plt.show()

with open(path+"/GPUCB_Learner_sliding_ex-1") as fp:
        ex_reward=json.load(fp)
with open(path+"/GPUCB_Learner_sliding-1") as fp:
        reward=json.load(fp)

plt.figure()
plt.plot(ex_reward[0])
plt.plot(reward[0])
plt.legend(["exact reward","reward"])
plt.show()