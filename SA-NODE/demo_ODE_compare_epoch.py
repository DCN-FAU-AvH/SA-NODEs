#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

from models.dataloaders import ODE_Dataset
from models.training import Trainer
from models.neural_odes import NeuralODE
from models.sa_nodes import SANODE
from models.plot import plot_epoch_error
from models.utils import calculate_error_ODE
import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # =============================================================================
    # DATA
    # =============================================================================
    ode_type = 'nonlinear_nonautonomous'
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
    reg = 'l1' 
    # =============================================================================
    # TRADITIONAL NEURAL ODE MODEL
    # ======================================================
    node = NeuralODE(device, 
                    data_dim, 
                    hidden_dim, 
                    output_dim=2,
                    non_linearity='relu',
                    adjoint=False,
                    T=T, 
                    time_steps=time_steps).to(device)
    # ======================================================
    # OPTIMIZER
    # ======================================================
    optimizer_node = torch.optim.Adam(node.parameters(), lr=learning_rate)
    scheduler_node = lr_scheduler.StepLR(optimizer_node, step_size=1000, gamma=0.8)
    # ======================================================
    # TRAINING
    # ======================================================
    trainer_node = Trainer(node, 
                            optimizer_node, 
                            scheduler_node,
                            device, 
                            reg=reg)
    print('Training traditional neural ODE model:')
    trainer_node.train(u_train, num_epochs)
    # =============================================================================
    # SEMI-AUTONOMOUS NEURAL ODE MODEL
    # ======================================================
    reg = 'barron' # 'l1' or 'barron'
    sanode = SANODE(device, 
                    data_dim, 
                    hidden_dim, 
                    output_dim=2,
                    non_linearity='relu',
                    adjoint=False,
                    T=T, 
                    time_steps=time_steps).to(device)
    # ======================================================
    # OPTIMIZER
    # ======================================================
    optimizer_semi = torch.optim.Adam(sanode.parameters(), lr=learning_rate)
    scheduler_semi = lr_scheduler.StepLR(optimizer_semi, step_size=1000, gamma=0.8)
    # ======================================================
    # TRAINING
    # ======================================================
    trainer_semi = Trainer(sanode, 
                            optimizer_semi, 
                            scheduler_semi,
                            device, 
                            reg=reg)
    print('Training SA-NODE model:')
    trainer_semi.train(u_train, num_epochs)
    # =============================================================================
    # TEST
    # =============================================================================
    u_test_init = u_test[0, :, :]
    # traditional neural ODE
    # test dataset
    u_pred_trad, traj_pred_trad = node(u_test_init)
    u_test_trad = torch.cat((u_test_init.unsqueeze(0), traj_pred_trad), dim=0)
    u_test_trad = u_test_trad.detach().cpu().numpy()
    # train dataset
    u_train_init = u_train[0, :, :]
    u_pred_train_trad, traj_pred_train_trad = node(u_train_init)
    u_train_trad = torch.cat((u_train_init.unsqueeze(0), traj_pred_train_trad), dim=0)
    u_train_trad = u_train_trad.detach().cpu().numpy()

    # semi-neural ODE
    # test dataset
    u_pred_semi, traj_pred_semi = sanode(u_test_init)
    u_test_semi = torch.cat((u_test_init.unsqueeze(0), traj_pred_semi), dim=0)
    u_test_semi = u_test_semi.detach().cpu().numpy()
    # train dataset
    u_pred_train_semi, traj_pred_train_semi = sanode(u_train_init)
    u_train_semi = torch.cat((u_train_init.unsqueeze(0), traj_pred_train_semi), dim=0)
    u_train_semi = u_train_semi.detach().cpu().numpy()

    # Exact solution
    u_train = u_train.detach().cpu().numpy()
    u_test = u_test.detach().cpu().numpy()
    u_real = np.concatenate((u_train, u_test), axis=1)

    # calculate the error
    error_trad, error_trad_max, error_trad_end = calculate_error_ODE(u_test, u_test_trad)
    error_semi, error_semi_max, error_semi_end = calculate_error_ODE(u_test, u_test_semi)

    return trainer_node.epoch_record, trainer_node.error_record, trainer_semi.error_record
    

if __name__ == '__main__':
    for num_epochs in [10000]:
        for hidden_dim in [1000]:
            epoch_list, error_trad_max, error_semi_max = main()
    
    # plot the results
    plot_epoch_error(epoch_list, error_semi_max, error_trad_max)
    
    
    
    



