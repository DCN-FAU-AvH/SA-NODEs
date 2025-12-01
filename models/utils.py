import numpy as np
import torch
import gc
from torch import sin, cos, cosh, tanh, sqrt, zeros_like
from torchdiffeq import odeint

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_error_ODE(u_real, u_NN):
    error_x = u_real[:, :, 0] - u_NN[:, :, 0]
    error_y = u_real[:, :, 1] - u_NN[:, :, 1]
    error = np.sqrt(error_x ** 2 + error_y ** 2) 

    error_linfty = error.mean(axis=1).max() 
    error_T = error.mean(axis=1)[-1]  

    return error, error_linfty, error_T

def sech(x):
    # use torch.cosh, not math.cosh!
    return 1.0 / cosh(x)

def model(u, t, transport_type):
    # u: [..., 3], t: scalar tensor
    x, y, z = u[..., 0], u[..., 1], u[..., 2]

    if transport_type == 'nonautonomous':
        denom = 1.0 + t**2
        dx = sin(x) / denom
        dy = sin(y) / denom
        dz = -z * (cos(x) + cos(y)) / denom

    elif transport_type == 'Doswell':
        r = sqrt(x**2 + y**2) + 1e-10
        common = 2.58907 * sech(r)**2 * tanh(r) / r
        dx = -y * common
        dy =  x * common
        dz = zeros_like(z)

    return torch.stack([dx, dy, dz], dim=-1)

def init_cond(XY, transport_type, mode):
    # XY: [N,2], already on the correct device
    if transport_type == 'nonautonomous':
        if mode == 'train':
            u_init = torch.full((XY.shape[0], 1), 0.5, device=XY.device)
        else:  # 'test'
            sigma = 2.0
            exponent = -torch.sum(XY**2, dim=1) / (sigma**2)
            u_init = torch.exp(exponent).unsqueeze(-1)

    else:  # 'Doswell'
        sigma = 1.0 if mode == 'train' else 0.1
        u_init = torch.tanh(XY[:, 1] / sigma).unsqueeze(-1)

    # concatenate XY (already on device) with u_init (also on device)
    return torch.cat([XY, u_init], dim=1)  # shape [N,3]

def calculate_error_transport(t, anode, dt, transport_type, mode):
    # Build grid
    l = 4.0 if transport_type=='nonautonomous' else 5.0
    N_x = 201
    x = torch.linspace(-l, l, N_x, device=device)
    y = torch.linspace(-l, l, N_x, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]

    num_fig = t.shape[0]
    step = (num_fig - 1) // (num_fig - 1)  # i.e., 1
    time_indices = list(range(1, num_fig, step))
    t_step = np.linspace(0, t[-1].item(), num_fig)

    # Allocate
    u_exact = torch.zeros(num_fig, N_x, N_x)
    u_NN    = torch.zeros_like(u_exact)

    # Initial field at t=0
    u0 = init_cond(XY, transport_type, mode).to(device)  # [N,3]
    initial_img = u0[:,2].reshape(N_x, N_x).cpu()
    u_exact[0] = initial_img
    u_NN[0]    = initial_img

    for idx, ti in enumerate(time_indices, start=1):
        # backward and forward time grids
        t_bwd = torch.linspace(ti*dt, 0.0, ti+1, device=device)
        t_fwd = torch.linspace(0.0, ti*dt, ti+1, device=device)

        # --- Exact solution: backward then forward ---
        with torch.no_grad():
            u0_bwd = torch.cat([XY, torch.zeros(XY.shape[0],1, device=device)], dim=1)
            sol_bwd = odeint(lambda tt, yy: model(yy, tt, transport_type),
                              u0_bwd, t_bwd, method='euler')
            u_back  = sol_bwd[-1]

            u0_fwd = init_cond(u_back[:, :2], transport_type, mode).to(device)
            sol_fwd = odeint(lambda tt, yy: model(yy, tt, transport_type),
                              u0_fwd, t_fwd, method='euler')
            u_exact[idx] = sol_fwd[-1,:,2].reshape(N_x, N_x).cpu()

        # --- SA-NODE solution: forward only ---
        with torch.no_grad():
            u0_bwd = torch.cat([XY, torch.zeros(XY.shape[0],1, device=device)], dim=1)
            sol_nn = anode.integrate(
                u0_bwd, backward=0, eval_times=t_bwd
            )
            u_back = sol_nn[-1]

            u0_fwd = init_cond(u_back[:, :2], transport_type, mode).to(device)
            sol_nn = anode.integrate(
                u0_fwd, backward=0, eval_times=t_fwd
            )
            u_NN[idx] = sol_nn[-1,:,2].reshape(N_x, N_x).cpu()

        torch.cuda.empty_cache()
        gc.collect()

    # Compute relative L1 error
    error_rel = np.zeros(num_fig)
    for i in range(num_fig):
        e = u_exact[i].numpy()
        n = u_NN[i].numpy()
        error_rel[i] = np.sum(np.abs(e - n)) / np.sum(np.abs(e))

    return t_step, error_rel