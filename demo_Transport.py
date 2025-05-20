#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

from models.dataloaders import Transport_Dataset
from models.training import Trainer
from models.sa_nodes import SANODE_transport
from models.plot import plot_transport, plot_transport_error
from models.utils import calculate_error_transport
import torch
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # =============================================================================
    # DATA
    # =============================================================================
    transport_type='Doswell'      # 'nonautonomous' or 'Doswell'
    datasets = Transport_Dataset(transport_type, mode='train')
    t, u = datasets[:]
    # separate the data into u_train and u_test
    u_train = u.to(device)
    # =============================================================================
    # SETUP max(t).item()
    # =============================================================================
    T, num_steps = max(t).item(), datasets.length_t()
    dt = T/(num_steps-1)
    data_dim = u.shape[2]-1

    # hyperparameters for the model
    time_steps = num_steps
    learning_rate = 1e-3
    reg = 'barron' # 'l1' or 'barron'
    # =============================================================================
    # MODEL
    # =============================================================================
    anode = SANODE_transport(device, 
                    data_dim, 
                    hidden_dim, 
                    output_dim=2,
                    non_linearity='sigmoid',
                    adjoint=False,
                    T=T, 
                    time_steps=time_steps).to(device)
    # =============================================================================
    # OPTIMIZER
    # =============================================================================
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_anode, step_size=100000, gamma=0.8)
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_anode = Trainer(anode, 
                            optimizer_anode, 
                            scheduler,
                            device, 
                            reg=reg)

    trainer_anode.train(u_train, num_epochs)
    torch.save(anode.state_dict(), 'transport_sanode.pt')
    # =============================================================================
    # TEST
    # =============================================================================
    # train dataset
    u_train_init = u_train[0, :, :]
    u_pred_train, traj_pred_train = anode(u_train_init, backward=0)
    # combine the initial condition and the predicted trajectory
    u_pred_save_train = torch.cat((u_train_init.unsqueeze(0), traj_pred_train), dim=0)
    u_train_NN = u_pred_save_train.detach().cpu().numpy()
    u_train = u_train.detach().cpu().numpy()
    
    # calculate the error
    error_train = calculate_error_transport(u_train, u_train_NN)


    # load test data
    datasets = Transport_Dataset(transport_type, mode='test')
    t, u = datasets[:]
    # separate the data into u_train and u_test
    u_test = u.to(device)
    # test dataset
    u_test_init = u_test[0, :, :]
    u_pred_test, traj_pred_test = anode(u_test_init, backward=0)
    # combine the initial condition and the predicted trajectory
    u_pred_save_test = torch.cat((u_test_init.unsqueeze(0), traj_pred_test), dim=0)
    u_test_NN = u_pred_save_test.detach().cpu().numpy()
    u_test = u_test.detach().cpu().numpy()
    
    # calculate the error
    error_test = calculate_error_transport(u_test, u_test_NN)

    # plot the results
    plot_transport_error(t, error_train, error_test)
    plot_transport(t, anode, dt, transport_type)
    

if __name__ == '__main__':
    for num_epochs in [20000]:
        for hidden_dim in [200]:
            main()
    
    
    
    



