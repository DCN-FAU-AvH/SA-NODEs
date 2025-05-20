#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

import torch.nn as nn
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
import time
from models.utils import calculate_error_ODE

console = Console(width=100)

losses = {'mse': nn.MSELoss(), 
          'cross_entropy': nn.CrossEntropyLoss(), 
          'ell1': nn.SmoothL1Loss(),
          'multi-margin': nn.MultiMarginLoss()
}

class Trainer():

    def __init__(self, 
                 model, 
                 optimizer, 
                 scheduler,
                 device, 
                 reg='l1',
                 print_freq=100, 
                 record_freq=100, 
                 save_dir=None):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reg = reg
        self.device = device
        self.loss_func = losses['mse']
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.error_record = []
        self.epoch_record = []


    def train(self, u, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = time.time()
        with Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True
            ) as progress:
            
            for epoch in progress.track(range(num_epochs)):
                loss = self._train_epoch(u)

                if (epoch+1)%self.print_freq==0:
                    console.print("Epoch {}/{} | Loss: {:.3e} | Time: {:.1f} min"\
                        .format(epoch+1, self.num_epochs, loss, (time.time()-self.start_time)/60.))
                    
                if (epoch+1)%self.record_freq==0:
                    u_pred, traj = self.model(u[0, :, :])
                    error, error_linfty, error_T = calculate_error_ODE(u[1:, :, :].cpu().numpy(), traj.cpu().detach().numpy())
                    self.error_record.append((error_linfty))
                    self.epoch_record.append(epoch+1)
        
        # print the final time in secs
        console.print("Total time: {:.1f} sec".format(time.time()-self.start_time))


    def _train_epoch(self, u):

        # Divide the data into initial and target 
        u_train = u[0, :, :].requires_grad_(True)
        u_target = u[1:, :, :].requires_grad_(True)
        
        u_pred, traj = self.model(u_train)
        
        ls_reg = 0.
        if self.reg == 'l1':
            for param in self.model.parameters():
                ls_reg += param.abs().sum()
            lambda_reg = 1e-4
        elif self.reg == 'barron':
            ls_reg = ((self.model.fc1_time_1.weight**2+self.model.fc1_time_2.weight**2).sum(1).sqrt() * \
                    (self.model.fc3_time_1.weight**2+self.model.fc3_time_2.weight**2).t().sum(1).sqrt()).sum()
            lambda_reg = 1e-4

        loss = 100*self.loss_func(traj[:,:,0:2], u_target[:,:,0:2]) + lambda_reg*ls_reg
        
        # for transport equations
        if u_target.shape[2] > 2:
            loss += 10000*self.loss_func(traj[:,:,2], u_target[:,:,2])
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()          
        self.scheduler.step()
        
        return loss.item()
                    