# Description: This file contains utility functions for the models.
import numpy as np
from scipy.interpolate import griddata
import torch
import gc
from math import sin, cos, tanh, sqrt, cosh
from scipy.integrate import odeint

def calculate_error_ODE(u_real, u_NN):
    # u_real and u_NN have the same shape [100, 64, 2]
    # error should be [100, 64], only axis=2 take square of the error
    error_x = u_real[:, :, 0] - u_NN[:, :, 0]
    error_y = u_real[:, :, 1] - u_NN[:, :, 1]
    error = np.sqrt(error_x ** 2 + error_y ** 2)  # shape [100, 64]

    # error_mean = error.mean(axis=1)  # shape [100]
    error_linfty = error.mean(axis=1).max() 
    error_T = error.mean(axis=1)[-1]  # shape [100]

    return error, error_linfty, error_T

def calculate_error_transport(t, anode, dt, transport_type, mode):

    # Choose plotting parameters based on transport_type
    if transport_type == 'nonautonomous':
        l = 4
    elif transport_type == 'Doswell':
        l = 5

    # Build the spatial grid
    N_x = 101
    x = np.linspace(-l, l, N_x)
    y = np.linspace(-l, l, N_x)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # Determine which time indices to sample
    num_fig = t.shape[0]
    step = (t.shape[0]-1) // (num_fig-1)
    time_indices = list(range(step, t.shape[0], step))
    t_step = np.linspace(0, t[-1].item(), num_fig)
    u_exact = np.zeros((num_fig, N_x, N_x))
    u_NN = np.zeros((num_fig, N_x, N_x))

    u_exact[0, :, :] = init_cond(XY_tensor, transport_type, mode)[:,-1].reshape(N_x, N_x).cpu().numpy()
    u_NN[0, :, :] = init_cond(XY_tensor, transport_type, mode)[:,-1].reshape(N_x, N_x).cpu().numpy()

    save_idx = 1
    for time in time_indices:
        # Build time arrays for backward and forward integration
        t_fwd = torch.linspace(0., time*dt, time+1)
        t_bwd = torch.linspace(time*dt, 0., time+1)

         # --- Exact solution: backward then forward ---
        with torch.no_grad():
            # Backward integration on the true model
            u0_bwd = torch.cat([XY_tensor, torch.zeros(XY_tensor.shape[0], 1)], dim=1).cpu()
            sol_bwd = batch_odeint(lambda y, τ: model(y, τ, transport_type),
                                   u0_bwd, t_bwd, batch_size=256)
            u_back = sol_bwd[-1]  # [N, 3]

            # Forward integration from mapped points
            XY_mapped = u_back[:, :2]
            u0_fwd = init_cond(XY_mapped, transport_type, mode)
            sol_fwd = batch_odeint(lambda y, τ: model(y, τ, transport_type),
                                   u0_fwd, t_fwd, batch_size=256)
            exact_img = sol_fwd[-1, :, 2].reshape(N_x, N_x).cpu().numpy()
            u_exact[save_idx, :, :] = exact_img

        # --- SA-NODE solution: backward then forward ---
        with torch.no_grad():
            sol_fwd_nn = anode.integrate(u0_fwd.to(anode.device), backward=0, eval_times=t_fwd.to(anode.device))
            nn_img = sol_fwd_nn[-1, :, 2].reshape(N_x, N_x).cpu().numpy()
            u_NN[save_idx, :, :] = nn_img

        # Cleanup
        del sol_bwd, sol_fwd, u_back, u0_bwd, sol_fwd_nn
        torch.cuda.empty_cache(); gc.collect()
        save_idx += 1

    # Compute the relative l1 error
    error_rel = np.zeros(num_fig)
    for i in range(num_fig):
        error_rel[i] = np.sum(np.abs(u_exact[i, :, :] - u_NN[i, :, :])) / np.sum(np.abs(u_exact[i, :, :]))


    return t_step, error_rel

def model(u0, t, transport_type):
    x, y, z = u0
    if transport_type == 'nonautonomous':
        dydt = [(sin(x))/(1+t**2), (sin(y))/(1+t**2), -z*(cos(x)+cos(y))/(1+t**2)]
    elif transport_type == 'Doswell':
        r = sqrt(x**2+y**2)+1e-10
        dydt = [-y*(1./r)*2.58907*(sech(r)**2)*tanh(r), x*(1./r)*2.58907*(sech(r)**2)*tanh(r),0]
    return dydt

def init_cond(XY, transport_type, mode):
    if transport_type == 'nonautonomous':
        if mode == 'train':
            u_init = torch.ones(XY.shape[0], 1) * 0.5
        elif mode == 'test':
            sigma = 2
            u_init = torch.exp(-torch.sum(XY**2, dim=1)/(sigma**2)).reshape(-1, 1)  
    elif transport_type == 'Doswell':
        sigma = 1.0 if mode == 'train' else 0.1
        u_init = torch.tanh(XY[:, 1]/sigma).reshape(-1, 1)
    u_init = torch.cat([XY, u_init], dim=1)
    return u_init

def sech(x):
    return 1 / cosh(x)

def batch_odeint(func, u0, t, batch_size=256):
    """
    A batch ODE solver that uses SciPy's odeint.
    
    Parameters:
      func: The function defining the ODE, should accept a 1D numpy array (converted to torch.Tensor) and a scalar time.
            It is expected to return a torch.Tensor or a list.
      u0: A torch.Tensor of shape [N, dim] representing the initial conditions.
      t: A torch.Tensor representing the 1D time grid.
      batch_size: The batch size to process the initial conditions.
      
    Returns:
      A torch.Tensor containing the solution with shape [time_steps, N, dim].
    """
    results = []
    n = u0.shape[0]
    # Convert time tensor to numpy array once for odeint
    t_np = t.cpu().numpy()
    
    for i in range(0, n, batch_size):
        batch_u0 = u0[i:i+batch_size]
        batch_sol_list = []
        # Process each initial condition individually
        for j in range(batch_u0.shape[0]):
            # Convert the j-th initial condition to a 1D numpy array
            y0 = batch_u0[j].cpu().numpy().flatten()
            # Define a wrapper function for odeint
            def f(y, t_scalar):
                # Convert y back into a torch.Tensor for model processing
                y_tensor = torch.tensor(y, dtype=torch.float32)
                # Call the provided function (e.g., a lambda wrapping model) with y_tensor and time t_scalar
                y_out = func(y_tensor, t_scalar)
                # If y_out is a list, convert it to a torch.Tensor
                if isinstance(y_out, list):
                    y_out = torch.tensor(y_out, dtype=torch.float32)
                # Return a flattened numpy array (the odeint solver expects a 1D array)
                return y_out.cpu().numpy().flatten()
            sol = odeint(f, y0, t_np)
            batch_sol_list.append(sol)
        # Stack solutions along the batch dimension. The resulting shape will be (time_steps, batch_size, dim)
        batch_sol = np.stack(batch_sol_list, axis=1)
        # Convert the batch result back to a torch.Tensor
        batch_sol = torch.tensor(batch_sol, dtype=torch.float32)
        results.append(batch_sol)
        # Free up memory
        del batch_u0, batch_sol_list, batch_sol
        torch.cuda.empty_cache()
        gc.collect()
    
    # Concatenate all batches along the batch dimension (axis=1) and return
    return torch.cat(results, dim=1)

# def calculate_error_transport(u_real: np.ndarray,
#                               u_NN:   np.ndarray,
#                               nx:     int    = 100,
#                               ny:     int    = 100,
#                               method: str    = 'cubic',
#                               fill_value: float = 0.0,
#                               eps:    float = 1e-12) -> np.ndarray:
#     """
#     Compute the **relative** L¹ error of the rho-field between two trajectory datasets
#     at each time step.

#     At each time t, we approximate
#         error_L1(t) = ∫ |ρ_real(t,x,y) − ρ_NN(t,x,y)| dx dy
#         norm_L1(t)  = ∫ |ρ_real(t,x,y)|                dx dy
#         rel_error(t) = error_L1(t) / (norm_L1(t) + eps)

#     by interpolating each scattered (x,y,ρ) dataset onto an nx×ny grid.

#     Parameters
#     ----------
#     u_real : ndarray, shape (N_t, N_x, 3)
#         Ground-truth trajectories and densities: each entry is (x, y, rho).
#     u_NN : ndarray, shape (N_t, N_x, 3)
#         Predicted trajectories and densities from the neural network.
#     nx, ny : int, optional
#         Grid resolution in x and y for interpolation (default: 80×80).
#     method : {'linear','nearest','cubic'}, optional
#         Interpolation method passed to SciPy’s griddata (default: 'linear').
#     fill_value : float, optional
#         Value outside convex hull (default: 0.0).
#     eps : float, optional
#         Small term to avoid division by zero (default: 1e-12).

#     Returns
#     -------
#     rel_error : ndarray, shape (N_t,)
#         The relative L¹ error at each time step.
#     """
#     N_t = u_real.shape[0]
#     rel_error = np.zeros(N_t)

#     for t in range(N_t):
#         # 1) Extract coords and rho at time t
#         coords_r = u_real[t, :, :2]  # shape (N_x, 2)
#         rho_r    = u_real[t, :,  2]  # shape (N_x,)
#         coords_p = u_NN[t,   :, :2]
#         rho_p    = u_NN[t,   :,  2]

#         # 2) Flatten is trivial here since we already have (N_x, ...)
#         X1, y1 = coords_r, rho_r
#         X2, y2 = coords_p, rho_p

#         # 3) Bounding box
#         all_pts = np.vstack((X1, X2))
#         xmin = X1[0, 0].item()
#         ymin = X1[0, 0].item()
#         xmax = -xmin
#         ymax = -ymin

#         # 4) Build grid
#         xs = np.linspace(xmin, xmax, nx)
#         ys = np.linspace(ymin, ymax, ny)
#         xx, yy = np.meshgrid(xs, ys)
#         grid = np.column_stack((xx.ravel(), yy.ravel()))

#         # 5) Interpolate
#         f_grid = griddata(X1, y1, grid, method=method, fill_value=fill_value)
#         g_grid = griddata(X2, y2, grid, method=method, fill_value=fill_value)

#         # 6) Compute L1 on grid
#         diff = np.abs(f_grid - g_grid)
#         area = (xmax - xmin) * (ymax - ymin)
#         cell_area = area / (nx * ny)
#         err_L1 = diff.sum() * cell_area

#         # 7) Compute L1 norm of real field on same grid
#         norm_L1 = np.abs(f_grid).sum() * cell_area

#         # 8) Relative error
#         rel_error[t] = err_L1 / (norm_L1 + eps)

#     return rel_error