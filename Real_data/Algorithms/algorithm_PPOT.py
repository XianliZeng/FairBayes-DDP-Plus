import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
import postprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DataLoader.dataloader import CustomDataset
import time

from DataLoader.dataloader import FairnessDataset
import torch.optim as optim
from models import Classifier
def get_level(dataset_name):
    if dataset_name == 'AdultCensus':
        level_list = np.arange(10)/50


    return level_list

def PPOT(dataset,dataset_name,net, optimizer, lr_schedule, device, n_epochs=200, batch_size=2048, seed=0):
    training_tensors,validation_tensors, testing_tensors = dataset.get_dataset_in_tensor(seed)
    X_train, Y_train, Z_train, XZ_train = training_tensors
    X_val, Y_val, Z_val, XZ_val = validation_tensors
    X_test, Y_test, Z_test, XZ_test = testing_tensors

    # training data size and validation data size


    Y_val_np = Y_val.detach().cpu().numpy()

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

    eta_test = net(XZ_test).detach().cpu().numpy().squeeze()
    train_test_size = len(eta_train) + len(eta_test)

    probas_ = np.zeros((train_test_size,2))
    probas_[:,1] = np.concatenate((eta_train,eta_test))
    probas_[:, 0] = 1-np.concatenate((eta_train, eta_test))
    alphas = get_level(dataset_name)
    alphas[-1]=np.inf
    Y_train_np = Y_train.clone().detach().numpy()
    Z_train_np = Z_train.clone().detach().numpy()
    Z_train_np = ((Z_train_np == 1) * 1).astype(np.int64)

    Y_test_np = Y_test.clone().detach().numpy()
    Z_test_np = Z_test.clone().detach().numpy()
    Z_test_np = ((Z_test_np == 1) * 1).astype(np.int64)

    labels_ = np.concatenate((Y_train_np,Y_test_np))
    groups_ = np.concatenate((Z_train_np,Z_test_np))
    n_post = len(Y_train_np)
    n_test = len(Y_test_np)
    df_test = pd.DataFrame()
    for alpha in alphas:
        temp =  utils.postprocess( alpha_seed_and_kwargs = [alpha,seed],postprocessor_factory = postprocess.PostProcessorDP,  probas = probas_,labels = labels_,groups = groups_,n_post = n_post,n_test = n_test, dataset_name = dataset_name)
        df_test = pd.concat([df_test,temp])



    return df_test



def get_training_parameters(dataset_name):
    if dataset_name == 'AdultCensus':
        n_epochs = 200
        lr = 1e-1
        batch_size = 512

    return n_epochs,lr,batch_size






def training_PPOT(dataset_name,seed):
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
    Result = PPOT(dataset=dataset,dataset_name=dataset_name,
                     net=net,
                     optimizer=optimizer,lr_schedule=lr_schedule,
                     device=device, n_epochs=n_epochs, batch_size=batch_size, seed=seed)



    Result.to_csv(f'Result/PPOT/result_of_{dataset_name}_with_seed_{seed}')










