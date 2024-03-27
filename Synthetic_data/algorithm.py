
import numpy as np
from dataloader import generate_dataset
from localreg import *
from utils import measures_from_Yhat_DemPa, solvet, misclssification_rate, threshold_DemPa
import pandas as pd


def Training(Delta_list ,training_obj,training_sample_list ,offlevels, Validation_size = 1000, Test_size = 1000,s1 = 0.2,s2 = 0.8,beta = 1,seed = 0):
    df_testall = pd.DataFrame()
    np.random.seed(seed)

    print(f'We are running simulation result for seed: {seed}, beta: {beta}')

    for Training_size in training_sample_list:
        Train_Dataset1, Train_Dataset0 = generate_dataset(Training_size,s1,s2,beta)
        Val_Dataset1, Val_Dataset0 = generate_dataset(Validation_size,s1,s2,beta)
        Test_Dataset1, Test_Dataset0 = generate_dataset(Test_size,s1,s2,beta)

        [X_train1, Y_train1] = Train_Dataset1
        [X_train0, Y_train0] = Train_Dataset0
        [X_val1, Y_val1] = Val_Dataset1
        [X_val0, Y_val0] = Val_Dataset0
        [X_test1, Y_test1] = Test_Dataset1
        [X_test0, Y_test0] = Test_Dataset0
        h_level_list = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
        # Retrieve train/test splitted pytorch tensors for index=split
        n1all, d = X_train1.shape
        n0all, d = X_train0.shape

        acclist1 = []
        acclist0 = []
        for num_h,h_level in enumerate(h_level_list):
            h1  = h_level * n1all ** (-1 / (2 * beta + d))
            h0  = h_level * n0all ** (-1 / (2 * beta + d))

            eta_val1 = localreg(X_train1, Y_train1, X_val1, degree=beta, kernel=rbf.gaussian, radius=h1)
            eta_val0 = localreg(X_train0, Y_train0, X_val0, degree=beta, kernel=rbf.gaussian, radius=h0)
            acc1 = np.mean((eta_val1 > 0.5) == Y_val1)
            acc0 = np.mean((eta_val0 > 0.5) == Y_val0)
            acclist1.append(acc1)
            acclist0.append(acc0)

        hlevel_opt1 = h_level_list[acclist1.index(max(acclist1))]
        hlevel_opt0 = h_level_list[acclist0.index(max(acclist0))]

        hopt1 = hlevel_opt1 ** (-1 / (2 * beta + d))
        hopt0 = hlevel_opt0 ** (-1 / (2 * beta + d))

        eta_train1 = localreg(X_train1, Y_train1, degree=beta, kernel=rbf.gaussian, radius=hopt1)
        eta_test1 = localreg(X_train1, Y_train1, X_test1, degree=beta, kernel=rbf.gaussian, radius=hopt1)
        eta_train0 = localreg(X_train0, Y_train0, degree=beta, kernel=rbf.gaussian, radius=hopt0)
        eta_test0 = localreg(X_train0, Y_train0, X_test0, degree=beta, kernel=rbf.gaussian, radius=hopt0)
        for off_level in offlevels:
            l1 =  off_level * n1all ** (-1 * beta / (2 * beta + d))
            l0 =  off_level * n0all ** (-1 * beta / (2 * beta + d))
            for delta in Delta_list:
                tstar = solvet(delta, s1, s2, beta)
                MCstar = misclssification_rate(tstar, s1, s2, beta)
                rn = 0.1/(np.log(Training_size))
                Deltan = 0.1/(np.log(np.log(Training_size)))
                [t1_DDPall, t0_DDPall, tau1, tau0,flag] = threshold_DemPa(eta_train1, eta_train0, l1,l0, rn,Deltan, delta=delta,pre_level=1e-5)
                Y1hatall = eta_test1 > t1_DDPall + l1
                Y0hatall = eta_test0 > t0_DDPall + l0
                boundaryindex1 = ((eta_test1 < t1_DDPall + l1) & (eta_test1 > t1_DDPall - l1))
                boundaryindex0 = ((eta_test0 < t0_DDPall + l0) & (eta_test0 > t0_DDPall - l0))

                Y1hatall[boundaryindex1] = np.random.binomial(1, tau1, boundaryindex1.shape)[boundaryindex1]
                Y0hatall[boundaryindex0] = np.random.binomial(1, tau0, boundaryindex0.shape)[boundaryindex0]

                acc, MC, DDP = measures_from_Yhat_DemPa(Y1hatall, Y0hatall, Y_test1, Y_test0)


                DE = MC - MCstar + tstar * (DDP - delta)
                data = [acc, DE, np.abs(DDP)]
                columns = ['acc','FairExRisk', 'DDP']
                tempall = pd.DataFrame([data], columns=columns)
                tempall['h_level1'] = hlevel_opt1
                tempall['h_level0'] = hlevel_opt0
                tempall['beta'] = beta
                tempall['Sample_size'] = Training_size
                tempall['flag'] = flag

                tempall['offset_level'] = off_level
                tempall['seed'] = seed
                tempall['delta'] = delta
                df_testall = pd.concat([df_testall, tempall])
    if training_obj == 'sample':
        df_testall.to_csv(f'Result/samplesize/Simulation_result_for_seed_{seed}')
    if training_obj == 'delta':
        df_testall.to_csv(f'Result/delta/Simulation_result_for_seed_{seed}')


