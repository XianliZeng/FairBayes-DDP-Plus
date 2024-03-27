import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
from utils import threshold_flipping, cal_acc_PPF, cal_disparity_PPF, postpreocessing_flipping
import time

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier
def get_level(dataset_name):
    if dataset_name == 'AdultCensus':
        level_list = np.arange(10)/50
        level_list[0]=0.001

    return level_list

def PPF(dataset,dataset_name,net, optimizer, lr_schedule,device, n_epochs=200, batch_size=2048, seed=0):
    training_tensors,validation_tensors, testing_tensors = dataset.get_dataset_in_tensor(seed)
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_val, Y_val, Z_val, XZ_val = validation_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    # training data size and validation data size


    Y_val_np = Y_val.detach().cpu().numpy()
    #
    Y_train_np = Y_train.detach().cpu().numpy()
    Z_train_np = Z_train.detach().cpu().numpy()

    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()

    #### if DDP<0, let Z=1-Z####
    dp0 = (Y_train_np[Z_train_np==1]).mean() - (Y_train_np[Z_train_np==0]).mean()
    if dp0<0:
        Z_train = 1 - Z_train
        Z_train_np = 1 - Z_train_np
        Z_test_np = 1 - Z_test_np
    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == 'full':
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)



    loss_function = nn.BCELoss()
    with tqdm(range(n_epochs)) as epochs:
        epochs.set_description(f"Training the classifier with dataset: {dataset_name}, seed: {seed}")
        for epoch in epochs:
            net.train()
            for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
                xz_batch, y_batch, z_batch = xz_batch.to(device), y_batch.to(device), z_batch.to(device)
                Yhat = net(xz_batch)

                # prediction loss
                cost = loss_function(Yhat.squeeze(), y_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()


                epochs.set_postfix(loss=cost.item())

            lr_schedule.step()
            ########choose the model with best performance on validation set###########
            with torch.no_grad():
                Yhat_val = net(XZ_val).detach().squeeze().numpy() > 0.5
                accuracy = (Yhat_val == Y_val_np).mean()

                if epoch == 0:
                    accuracy_max = accuracy
                    bestnet_acc_stat_dict = net.state_dict()

                if accuracy > accuracy_max:
                    accuracy_max = accuracy
                    bestnet_acc_stat_dict = net.state_dict()

    ##########performance of fair Classifier on test sets###########
    net.load_state_dict(bestnet_acc_stat_dict)

    eta_train = net(XZ_train).detach().cpu().numpy().squeeze()
    Y_hat_train = (eta_train>0.5)
    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    Y_hat_test = (eta_test>0.5)
    pa = torch.mean(Z_train).item()


    df_test = pd.DataFrame()

    level_list = get_level(dataset_name)
    for level in level_list:
        t = threshold_flipping(pa,eta_train, Y_hat_train,Y_train_np,Z_train_np,level)
        Y_hat_test_new =  postpreocessing_flipping(pa,eta_test,Y_hat_test,Z_test_np,t)
        acc = cal_acc_PPF(Y_hat_test_new,Y_test_np)
        disparity = cal_disparity_PPF(Y_hat_test_new,Z_test_np)
        data = [seed,dataset_name,level,acc, np.abs(disparity)]
        columns = ['seed','dataset','level','acc', 'disparity']


        temp_dp = pd.DataFrame([data], columns=columns)
        df_test= pd.concat([df_test,temp_dp])

    return df_test





def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512

    return n_epochs,lr,batch_size






def training_PPF(dataset_name,seed):
    device = torch.device('cpu')



    # Set a seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Import dataset
    dataset = FairnessDataset(dataset=dataset_name, seed = seed, device=device)
    dataset.normalize()
    input_dim = dataset.XZ_train.shape[1]
    n_epochs, lr, batch_size= get_training_parameters(dataset_name)
    # Create a classifier model
    net = Classifier(n_inputs=input_dim)
    net = net.to(device)

    # Set an optimizer
    lr_decay = 0.98
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    # Fair classifier training
    Result = PPF(dataset=dataset,dataset_name=dataset_name,
                     net=net,
                     optimizer=optimizer,lr_schedule=lr_schedule,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print(Result)
    Result.to_csv(f'Result/PPF/result_of_{dataset_name}_with_seed_{seed}')


