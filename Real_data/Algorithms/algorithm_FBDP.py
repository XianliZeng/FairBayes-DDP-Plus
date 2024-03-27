import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import cal_disparity,threshold_DemPa,cal_acc,measures_from_Yhat_DemPa

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier
from localreg import localreg,rbf

def FBDP(dataset,dataset_name,net, optimizer, lr_schedule, device, n_epochs=200, batch_size=2048, seed=0):



    h_level_list = [t/2+0.5 for t in range(10)]
    offset_level = 0.1
    # Retrieve train/test splitted pytorch tensors for index=split
    train_tensors, val_tensors,test_tensors = dataset.get_dataset_in_tensor()
    beta_list = [3]
    delta_list = [0.004 * t for t in range(50)]
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_val, Y_val, Z_val, XZ_val = train_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    train_size  = len(X_train)
    val_size = len(X_val)
    test_size =len(X_test)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    print(Y_test.mean())


    X_train1, X_train0 = X_train[Z_train == 1].clone().detach().cpu().numpy(), X_train[Z_train == 0].clone().detach().cpu().numpy()
    Y_train1, Y_train0 = Y_train[Z_train == 1].clone().detach().cpu().numpy(), Y_train[Z_train == 0].clone().detach().cpu().numpy()

    X_val1, X_val0 = X_val[Z_val == 1].clone().detach().cpu().numpy(), X_val[Z_val == 0].clone().detach().cpu().numpy()
    Y_val1, Y_val0 = Y_val[Z_val == 1].clone().detach().cpu().numpy(), Y_val[Z_val == 0].clone().detach().cpu().numpy()

    X_test1, X_test0 = X_test[Z_test == 1].clone().detach().cpu().numpy(), X_test[Z_test == 0].clone().detach().cpu().numpy()
    Y_test1, Y_test0 = Y_test[Z_test == 1].clone().detach().cpu().numpy(), Y_test[Z_test == 0].clone().detach().cpu().numpy()
    # training data size and validation data size

    # train_val_size = len(X_train_val)
    # Y_train,Y_val=np.split(Y_train_val,int(train_val_size*0.8))
    # Z_train,Z_val=np.split(Z_train_val,int(train_val_size*0.8))
    # X_train,X_val=np.split(X_train_val,int(train_val_size*0.8))
    n1all, d = X_train1.shape
    n0all, d = X_train0.shape
    df_test = pd.DataFrame()

    for beta in beta_list:

        print(f'sample size 1 is {n1all}, sample size 0 is {n0all}, dimension is {d}, seed is {seed}')
        acclist1 = []
        acclist0 = []
        for num_h, h_level in enumerate(h_level_list):
            h1 = h_level * n1all ** (-1 / (2 * beta + d))
            h0 = h_level * n0all ** (-1 / (2 * beta + d))

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

        l1 = offset_level * n1all ** (-1 * beta / (2 * beta + d))
        l0 = offset_level * n0all ** (-1 * beta / (2 * beta + d))

        for delta in delta_list:
            rn = 0.1 / (np.log(train_size))
            Deltan = 0.1 / (np.log(np.log(train_size)))
            [t1_DDPall, t0_DDPall, tau1, tau0, flag] = threshold_DemPa(eta_train1, eta_train0, l1, l0, rn, Deltan,
                                                                       delta=delta, pre_level=1e-5)
            Y1hatall = eta_test1 > t1_DDPall + l1
            Y0hatall = eta_test0 > t0_DDPall + l0
            boundaryindex1 = ((eta_test1 < t1_DDPall + l1) & (eta_test1 > t1_DDPall - l1))
            boundaryindex0 = ((eta_test0 < t0_DDPall + l0) & (eta_test0 > t0_DDPall - l0))

            Y1hatall[boundaryindex1] = np.random.binomial(1, tau1, boundaryindex1.shape)[boundaryindex1]
            Y0hatall[boundaryindex0] = np.random.binomial(1, tau0, boundaryindex0.shape)[boundaryindex0]

            acc,  DDP = measures_from_Yhat_DemPa(Y1hatall, Y0hatall, Y_test1, Y_test0)
            data = [seed,  hlevel_opt1,hlevel_opt0, offset_level, beta,acc,np.abs(DDP),delta,flag]
            columns = ['seed', 'hlevel_opt1','hlevel_opt0', 'off_level','beta','acc', 'disparity','delta','flag']
            tempall = pd.DataFrame([data], columns=columns)




            tempall['delta'] = delta
            df_test = pd.concat([df_test, tempall])
    return df_test




def get_training_parameters():
    n_epochs = 200
    lr = 1e-1
    batch_size = 512

    return n_epochs,lr,batch_size






def training_FBDP(dataset_name,seed):
    device = torch.device('cpu')

    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]
    n_epochs, lr, batch_size= get_training_parameters()
    # Create a classifier model
    net = Classifier(n_inputs=input_dim)
    net = net.to(device)
    # Set an optimizer and decay rate
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Fair classifier training

    Result = FBDP(dataset=dataset,dataset_name=dataset_name,
                     net=net,optimizer=optimizer,
                     lr_schedule=lr_schedule,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)

    Result.to_csv(f'Result/FBDP/result_of_{dataset_name}_with_seed_{seed}')










