

from joblib import Parallel, delayed
import numpy as np


from Algorithms.algorithm_KDE import training_KDE
from Algorithms.algorithm_FBDP import training_FBDP
from Algorithms.algorithm_ADV import training_ADV
from Algorithms.algorithm_PPF import training_PPF
from Algorithms.algorithm_PPOT import training_PPOT


lambda_list = np.arange(10) / 10 + 0.05
alphaA_list = np.arange(11)/2
def get_parameter_list(method):
    if method == 'KDE':
        parameter_list = lambda_list
    elif method == 'ADV':
        parameter_list = alphaA_list
    return parameter_list

def training(method,dataset_list,n_seeds,parallel_core_number):

    if method == 'KDE':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_KDE)(dataset, lambda_, seed) for lambda_ in parameter_list for seed in range(n_seeds))

    if method == 'ADV':
        for dataset in dataset_list:
            parameter_list = get_parameter_list(method)
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_ADV)(dataset, alpha, seed) for alpha in parameter_list for seed in range(n_seeds))

    if method == 'FBDP':

        for dataset in dataset_list:
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_FBDP)(dataset,  seed) for seed in range(n_seeds))

    if method == 'PPOT':
        for dataset in dataset_list:

            Parallel(n_jobs=parallel_core_number)(
                delayed(training_PPOT)(dataset,  seed) for seed in range(n_seeds))

    if method == 'PPF':
        for dataset in dataset_list:
            Parallel(n_jobs=parallel_core_number)(
                delayed(training_PPF)(dataset,  seed) for seed in range(n_seeds))

