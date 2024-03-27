
import IPython
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from algorithm import Training
import time
IPython.display.clear_output()




n_seeds = 1000 # Number of random seeds to try
##### Effects of disparity sample sizes#####
training_sample_list = [100,200,400,800,1600,3200,6400,12800]
beta_list = [1]
Delta_list=[0]
off_level_list = [0, 0.25, 0.5, 0.75, 1]

Parallel(n_jobs=20)(delayed(Training)(Delta_list = Delta_list,training_obj = 'sample',training_sample_list = training_sample_list,offlevels = off_level_list, Validation_size = 1000, Test_size = 10000,s1 = 0.2,s2 = 0.8,beta = 1,seed = seed) for seed in range(n_seeds))
t2 = time.time()


##### Effect of disparity levels#####

training_sample_list = [12800]
beta_list = [1]
Delta_list=[0,0.05,0.10,0.15,0.20,0.25,0.30]
off_level_list = [0.25]

Parallel(n_jobs=20)(delayed(Training)(Delta_list = Delta_list,training_obj = 'delta',training_sample_list = training_sample_list, offlevels = off_level_list,Validation_size = 1000, Test_size = 10000,s1 = 0.2,s2 = 0.8,beta = 1,seed = seed) for seed in range(n_seeds))


####merge results######
result = pd.DataFrame()
n_seeds = 1000
for seed in range(n_seeds):
    tempresult = pd.read_csv(f"Result/samplesize/Simulation_result_for_seed_{seed}")
    result = pd.concat([result,tempresult])
result.to_csv(f'Result/Result_after_merge/sample_size_all')


result = pd.DataFrame()
n_seeds = 1000
for seed in range(n_seeds):
    tempresult = pd.read_csv(f"Result/delta/Simulation_result_for_seed_{seed}")
    result = pd.concat([result,tempresult])
result.to_csv(f'Result/Result_after_merge/delta_all')