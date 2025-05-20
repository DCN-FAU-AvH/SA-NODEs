#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ziqian Li
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.integrate import solve_ivp

class ODE_Dataset(Dataset):
    def __init__(self, type = 'nonlinear_autonomous', T=5, N=101, step=0.2, grid_max=2):
        """
        Initializes the dataset by simulating the ODE trajectories.

        Parameters:
            type (str): Type of ODE system to simulate. 
            T (float): End time for the simulation.
            N (int): Number of time points.
            step (float): Spacing for the grid of initial conditions in both x and y.
            grid_max (float): Grid spans from -grid_max to grid_max for both dimensions.
        """
        self.type = type
        # Create time vector (shape: [N])
        t = np.linspace(0, T, N)
        self.t = torch.tensor(t, dtype=torch.float32)

        # Generate grid of initial conditions over [-grid_max, grid_max] x [-grid_max, grid_max]
        x_vals = np.arange(-grid_max, grid_max + step/2, step)
        y_vals = np.arange(-grid_max, grid_max + step/2, step)
        
        # Create 2D meshgrid and flatten to obtain a list of initial conditions
        X, Y = np.meshgrid(x_vals, y_vals)
        xx = X.ravel()
        yy = Y.ravel()
        num_conditions = xx.size

        # Preallocate array to store trajectories.
        # Shape: (N time points, num_conditions, 2 state dimensions)
        u = np.zeros((N, num_conditions, 2))

        # Simulate the ODE for each initial condition using scipy's solve_ivp.
        for i in range(num_conditions):
            # initial condition is [x, y]
            sol = solve_ivp(self._ode_system, [0, T], [xx[i], yy[i]], t_eval=t, method='RK45')
            # sol.y has shape (2, N); transpose to (N, 2) and store
            u[:, i, :] = sol.y.T

        self.u = torch.tensor(u, dtype=torch.float32)

    def _ode_system(self, t, a):
        """
        Defines the ODE system:
            dx/dt = f1(x, y, t)
            dy/dt = f2(x, y, t)
        """
        if self.type == 'nonlinear_autonomous':
            return [a[1], -np.sin(a[0])]
        elif self.type == 'nonlinear_nonautonomous':
            return [a[1], a[0] - a[0]**3 + 0.1 * np.cos(np.pi * t)]
        elif self.type == 'linear_autonomous':
            return [a[1], -2*a[0]-3*a[1]]
        elif self.type == 'linear_nonautonomous':
            return [t-a[1], a[0]-t]

    def __getitem__(self, index):
        """
        Returns the data at a specific time index.

        Args:
            index (int): Index of the time point.

        Returns:
            tuple: (t[index], u[index, :, :])
                   - t[index] is the scalar time at the index.
                   - u[index, :, :] is a tensor of shape (num_conditions, 2) with the state for all trajectories.
        """
        return self.t[index], self.u[index, :, :]

    def __len__(self):
        return len(self.t)

    def length_t(self):
        """Returns the number of time points."""
        return len(self.t)
    
    def length_u(self):
        """Returns the number of trajectories (initial conditions)."""
        return self.u.shape[1]


class Transport_Dataset(Dataset):
    """
    PyTorch Dataset for 2D transport equation solutions on a fixed (x, y) grid.
    Uses backward-then-forward integration to compute z(x, y, t) in Eulerian frame.
    Data is stored in shape (N_t, N_pts, 3), where the last dimension is [x, y, z].
    """
    def __init__(self, transport_type='nonautonomous', mode='train',
                 N_time=51, M_space=10):
        """
        Args:
            transport_type (str): 'nonautonomous' or 'Doswell'
            mode (str): 'train' or 'test'
            N_time (int): number of time points in [0, T]
            M_space (int): spatial resolution factor, so dx = 1/M_space
        """
        # 1. Problem parameters
        if transport_type == 'nonautonomous':
            sigma = 2.0
            L = 4.0
        elif transport_type == 'Doswell':
            sigma = 1.0 if mode == 'train' else 0.1
            L = 5.0
        else:
            raise ValueError(f"Unknown transport_type `{transport_type}`")

        self.transport_type = transport_type
        self.N_time = N_time
        self.T = 5.0

        # 2. Uniform time grid
        t_uniform = np.linspace(0.0, self.T, N_time)

        # 3. Fixed spatial grid points
        dx = 1.0 / M_space
        x_vals = np.arange(-L, L + dx, dx)
        y_vals = np.arange(-L, L + dx, dx)
        X, Y = np.meshgrid(x_vals, y_vals)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (N_pts, 2)
        N_pts = pts.shape[0]

        # 4. Prepare z-field array
        data = np.zeros((N_time, N_pts, 3), dtype=np.float32)

        # 5. Solver tolerances
        rtol, atol = 1e-5, 1e-8

        # 6. Compute z(x, y, t) by backward → forward integration per grid point
        for j, (xj, yj) in enumerate(pts):
            x0 = xj
            y0 = yj
            # 6.1 Compute initial z0 at (x0, y0)
            if transport_type == 'nonautonomous':
                if mode == 'test':
                    z0 = np.exp(-(x0**2 + y0**2)/(sigma**2))
                elif mode == 'train':
                    # uniform distribution
                    z0 = 0.5
            else:
                z0 = np.tanh(y0/sigma)

            # 6.3 Forward integrate full (x, y, z) from t=0 to t=T
            def ode_xyz(t, Y):
                x, y_, z = Y
                if transport_type == 'nonautonomous':
                    dxdt = np.sin(x)/(1+t**2)
                    dydt = np.sin(y_)/(1+t**2)
                    dzdt = -z*(np.cos(x)+np.cos(y_))/(1+t**2)
                    return [dxdt, dydt, dzdt]
                else:
                    r = np.sqrt(x**2+y_**2) + 1e-10
                    coef = 2.59807 * ((1/np.cosh(r))**2) * np.tanh(r) / r
                    return [-y_*coef, x*coef, 0.0]

            sol_forw = solve_ivp(
                ode_xyz, (0.0, self.T), [x0, y0, z0],
                t_eval=t_uniform, rtol=rtol, atol=atol
            )

            # store the solution
            data[:, j, 0] = sol_forw.y[0, :]  # x
            data[:, j, 1] = sol_forw.y[1, :]  # y
            data[:, j, 2] = sol_forw.y[2, :]  # z

        # 8. Convert to torch tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.t = torch.tensor(t_uniform, dtype=torch.float32)
        self.N_pts = N_pts

    def __len__(self):
        """Number of time samples."""
        return self.N_time

    def __getitem__(self, idx):
        """
        Returns:
            t_val (float): time at index idx
            snapshot (Tensor): shape (N_pts, 3), columns [x, y, z]
        """
        return self.t[idx], self.data[idx]
    
    def length_t(self):
        """Returns the number of time points."""
        return len(self.t)