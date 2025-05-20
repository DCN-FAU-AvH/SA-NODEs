#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

from models.dataloaders import ODE_Dataset
from models.training import Trainer
from models.sa_nodes import SANODE
from models.plot import plot_ODEs_error, plot_ODEs
from models.utils import calculate_error_ODE
import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # =============================================================================
    # DATA
    # =============================================================================
    ode_type = 'nonlinear_autonomous' 
    datasets = ODE_Dataset(ode_type)
    t, u = datasets[:]
    # separate the data into u_train and u_test
    train_size = int(0.5 * datasets.length_u())
    shuffled_indices = torch.randperm(datasets.length_u())
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    u_train = u[:, train_indices, :].to(device)
    u_test = u[:, test_indices, :].to(device)
    # =============================================================================
    # SETUP 
    # =============================================================================
    T, num_steps = max(t).item(), datasets.length_t()
    dt = T/(num_steps-1)
    data_dim = u.shape[2]

    # hyperparameters for the model
    time_steps = num_steps
    learning_rate = 1e-3
    reg = 'barron' # 'l1' or 'barron'
    # =============================================================================
    # MODEL
    # =============================================================================
    anode = SANODE(device, 
                    data_dim, 
                    hidden_dim, 
                    output_dim=2,
                    non_linearity='relu',
                    adjoint=False,
                    T=T, 
                    time_steps=time_steps).to(device)
    # =============================================================================
    # OPTIMIZER
    # =============================================================================
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_anode, step_size=1000, gamma=0.8)
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, 
                            optimizer_anode, 
                            scheduler,
                            device, 
                            reg=reg)

    trainer_anode.train(u_train, num_epochs)
    # =============================================================================
    # TEST
    # =============================================================================
    # test dataset
    u_test_init = u_test[0, :, :]
    u_pred, traj_pred = anode(u_test_init)
    # combine the initial condition and the predicted trajectory
    u_pred_save = torch.cat((u_test_init.unsqueeze(0), traj_pred), dim=0)
    u_test_NN = u_pred_save.detach().cpu().numpy()

    # train dataset
    u_train_init = u_train[0, :, :]
    u_pred_train, traj_pred_train = anode(u_train_init)
    # combine the initial condition and the predicted trajectory
    u_pred_save_train = torch.cat((u_train_init.unsqueeze(0), traj_pred_train), dim=0)
    u_train_NN = u_pred_save_train.detach().cpu().numpy()

    # combine u_train and u_test
    u_train = u_train.detach().cpu().numpy()
    u_test = u_test.detach().cpu().numpy()
    u_real = np.concatenate((u_train, u_test), axis=1)

    # plot the results
    plot_ODEs(u_real, u_train_NN, u_test_NN, ode_type)

    # calculate the error
    error_train, error_train_linfty, error_train_T = calculate_error_ODE(u_train, u_train_NN)
    error_test, error_test_linfty, error_test_T = calculate_error_ODE(u_test, u_test_NN)
    print('Max error:', '{:.2e}'.format(error_test_linfty))
    print('End error:', '{:.2e}'.format(error_test_T))

    plot_ODEs_error(t, error_train, error_test)



if __name__ == '__main__':
    for num_epochs in [10000]:
        for hidden_dim in [1000]:
            main()
    
    
    
    



