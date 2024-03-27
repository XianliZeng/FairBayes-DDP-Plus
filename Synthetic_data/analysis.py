import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



####simulation results for different sample sizes########
result = pd.DataFrame()
n_seeds = 1000

result = pd.read_csv(f'Result/Result_after_merge/sample_size_all')

sample_size = [np.log(2**t *100) for t in range (8)]
fig = plt.figure(figsize=(8,7))

for s in range(5):
    ddplist = [result[s::5][t::8]['FairExRisk'].mean() for t in range(8)]
    if s == 0:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.00 \cdot n_a^{-1/4}$  ')
    if s == 1:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.25 \cdot n_a^{-1/4}$  ')
    if s == 2:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.50 \cdot n_a^{-1/4}$  ')
    if s == 3:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.75 \cdot n_a^{-1/4}$  ')
    if s == 4:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 1.00 \cdot n_a^{-1/4}$  ')

plt.xlabel('Sample Size', fontsize = 20)
plt.ylabel(r'$d_E(\widehat{f}^{\,\text{PI}}_{\delta,n})$', fontsize = 20)

plt.legend( fontsize = 16)

plt.xticks([np.log(2**t *100) for t in range (8)], ['100','200','400','800','1600','3200','6400','12800'], fontsize = 20)
plt.ylim([-0.005,0.12])
plt.yticks([0,0.02,0.04,0.06,0.08,0.10],  fontsize = 20)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,7))

for s in range(5):
    ddplist = [result[s::5][t::8]['DDP'].mean() for t in range(8)]
    if s == 0:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.00 \cdot n_a^{-1/4}$  ')
    if s == 1:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.25 \cdot n_a^{-1/4}$  ')
    if s == 2:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.50 \cdot n_a^{-1/4}$  ')
    if s == 3:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 0.75 \cdot n_a^{-1/4}$  ')
    if s == 4:
        plt.plot(sample_size,ddplist, '.-.' ,ms =20, label=r'$\ell_{n,a} = 1.00 \cdot n_a^{-1/4}$  ')

# plt.text(8,0.035,r'$\times 100$', fontsize = 20)
plt.legend( fontsize = 16)
plt.xlabel('Sample Size', fontsize = 20)
plt.ylabel('DDP', fontsize = 20)

plt.xticks([np.log(2**t *100) for t in range (8)], ['100','200','400','800','1600','3200','6400','12800'], fontsize = 20)
plt.yticks(  fontsize = 20)
# plt.ylim(  [-0.05,0.12])
plt.tight_layout()
plt.show()



####simulation results for different disparity levels#######


import matplotlib.pyplot as plt

from algorithm import solvet,misclssification_rate
import numpy as np
import pandas as pd
result = pd.DataFrame()
n_seeds = 20
result = pd.read_csv(f'Result/Result_after_merge/delta_all')
fig = plt.figure(figsize=(8,7))


# plt.subplot(1,2,1)

Delta_list=[0,0.05,0.10,0.15,0.20,0.25,0.30]

tstarlist = []
accstarstarlist = []
for delta in Delta_list:
    tstar = solvet(delta,0.2,0.8,1)
    tstarlist.append(tstar)
    acc = 1- misclssification_rate(tstar,0.2,0.8,1)
    accstarstarlist.append(acc)

ddplist = [result[t::7]['DDP'].mean() for t in range(7)]

plt.plot(Delta_list,ddplist, '.-.' ,ms =20, label='FairBayes-DDP+')

x = np.arange(30)/100
y=x
plt.plot(x,y,'--',label = r'$y=x$')

plt.legend( fontsize = 18)
plt.xlabel(r'Disparity level: $\delta$', fontsize = 20)
plt.ylabel('DDP', fontsize = 20)

plt.xticks(fontsize = 20)
plt.yticks(  fontsize = 20)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8,7))


FRlist = [result[t::7]['FairExRisk'].mean() for t in range(7)]

plt.plot(Delta_list,FRlist, '.-.' ,ms =20, label='FairBayes-DDP+')



plt.xlabel(r'Disparity level: $\delta$', fontsize = 20)
plt.ylabel(r'$d_E(\widehat{f}^{\,\text{PI}}_{\delta,n})$', fontsize = 20)
plt.ylim(0,0.01)

# plt.text(8,0.04,r'$\times 100$', fontsize = 20)
plt.legend( fontsize = 18)
plt.xticks(  fontsize = 20)

plt.yticks([0,0.002,0.004,0.006,0.008] , fontsize = 20)
plt.tight_layout()

plt.show()
