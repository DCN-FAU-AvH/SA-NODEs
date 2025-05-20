#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

from models.dataloaders import Transport_Dataset
from models.training import Trainer
from models.neural_odes import NeuralODE_transport
from models.sa_nodes import SANODE_transport
from models.plot import plot_transport_compare, plot_transport_compare_error
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
    reg = 'l1' 
    # =============================================================================
    # TRADITIONAL NEURAL ODE MODEL
    # =============================================================================
    node = NeuralODE_transport(device, 
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
    optimizer_node = torch.optim.Adam(node.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_node, step_size=100000, gamma=0.8)
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_node = Trainer(node, 
                            optimizer_node, 
                            scheduler,
                            device, 
                            reg=reg)
    print('Training traditional neural ODE model:')
    trainer_node.train(u_train, num_epochs)
    torch.save(node.state_dict(), 'transport_node.pt')
    # =============================================================================
    # SEMI-AUTONOMOUS NEURAL ODE MODEL
    # =============================================================================
    reg = 'barron' # 'l1' or 'barron'
    sanode = SANODE_transport(device, 
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
    optimizer_sanode = torch.optim.Adam(sanode.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer_sanode, step_size=100000, gamma=0.8)
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer_sanode = Trainer(sanode, 
                            optimizer_sanode, 
                            scheduler,
                            device, 
                            reg=reg)
    print('Training SA-NODE model:')
    trainer_sanode.train(u_train, num_epochs)
    torch.save(sanode.state_dict(), 'transport_sanode.pt')
    # =============================================================================
    # TEST
    # =============================================================================
    mode = 'test'
    t_step, error_node = calculate_error_transport(t, node, dt, transport_type, mode)
    t_step, error_sanode = calculate_error_transport(t, sanode, dt, transport_type, mode)
    plot_transport_compare_error(t_step, error_node, error_sanode)

    # plot the results
    plot_transport_compare(t, node, sanode, dt, transport_type)
    
    

if __name__ == '__main__':
    for num_epochs in [20000]:
        for hidden_dim in [200]:
            main()
    
    
    
    



