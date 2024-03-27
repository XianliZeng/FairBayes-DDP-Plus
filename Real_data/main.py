import time

import IPython
import pandas as pd

from training_setting import training,get_parameter_list
# from analysis_all import drow_plot, print_table



dataset_list = ['AdultCensus']
parallel_core_number = 23
n_seeds = 50 # Number of random seeds to try
IPython.display.clear_output()

methods_to_train = ['KDE']
#

############training##########
for method in methods_to_train:
    training(method,dataset_list = dataset_list,n_seeds=n_seeds,parallel_core_number=parallel_core_number)

#
# ##########merge results########
#
# methods_to_train = ['PPOT','PPF','ADV']
for dataset in dataset_list:
    for method in methods_to_train:
        if method in ['FBDP','PPOT', 'PPF']:
            Result_all = pd.DataFrame()
            for seed in range(n_seeds):
                temp = pd.read_csv(f'Result/{method}/result_of_{dataset}_with_seed_{seed}')
                Result_all = pd.concat([Result_all,temp])
        else:
            parameter_list = get_parameter_list(method)
            Result_all = pd.DataFrame()
            for seed in range(n_seeds):
                for para in parameter_list:
                    temp = pd.read_csv(f'Result/{method}/result_of_{dataset}_with_seed_{seed}_para_{int(para * 1000)}')
                    Result_all = pd.concat([Result_all, temp])
        Result_all.to_csv(f'Result/Result_after_merge/All_result_of_{dataset}_training_by_{method}')

#
#
# ###Print tables#####
# print_table(proposed_methods, dataset_list)
#
#
# ###plot_results######
# drow_plot(dataset_list)