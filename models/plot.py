import matplotlib.pyplot as plt
import numpy as np  
import torch
from math import sin, cos, tanh, sqrt, cosh
from scipy.integrate import odeint
import gc

def plot_ODEs(u_real, u_train, u_test, ode_type):
    # plot u_train and u_test in one figure with different colors, and plot the true solution in another figure
    fig = plt.figure(figsize=(12, 5), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    x_min, x_max, y_min, y_max = XY_ODE_type(ode_type)
    # plot every trajectory
    for i in range(u_train.shape[1]):
        ax1.plot(u_train[:, i, 0], u_train[:, i, 1], 'r', linewidth=1.0)
    ax1.plot([],[], 'r', linewidth=1.0, label='Training Dataset')
    for i in range(u_test.shape[1]):
        ax1.plot(u_test[:, i, 0], u_test[:, i, 1], '#32cd32', linewidth=1.0)
    ax1.plot([],[], '#32cd32', linewidth=1.0, label='Testing Dataset')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('SA-NODEs')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right')


    # plot every trajectory
    for i in range(u_real.shape[1]):
        ax2.plot(u_real[:, i, 0], u_real[:, i, 1], 'b', linewidth=1.0)
    ax2.plot(0, 0, marker='o', markersize=1.0, color='b')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Exact')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    plt.savefig('./figures/ODE.png', bbox_inches='tight')
    plt.close()

def plot_ODEs_error(t, error_list_train, error_list_test):
    # plot the errors in error_list_train and error_list_test at different t in one figure with different colors
    # error_list_test need to plot std
    t = t.detach().cpu().numpy()
    error_train_mean = error_list_train.mean(axis=1)
    error_test_mean = error_list_test.mean(axis=1)

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, error_train_mean, 'r', linewidth=1.0)
    ax.plot(t, error_test_mean, 'b', linewidth=1.0)
    
    error_std = []
    for i in range(error_list_test.shape[0]):
        error_std.append(error_list_test[i,:].std())
    error_test_std = np.array(error_std)
    ax.fill_between(t, error_test_mean - error_test_std, error_test_mean + error_test_std,\
                        color='gray', alpha=0.35)
    

    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Errors of SA-NODEs')
    # legend
    ax.legend(['Training Dataset', 'Testing Dataset'])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.savefig('./figures/ODE_error.png', bbox_inches='tight')
    plt.close()

def plot_ODEs_compare(u_real, u_train_trad, u_test_trad, u_train_semi, u_test_semi, ode_type):
    # plot in three subplots
    fig = plt.figure(figsize=(18, 5), dpi=100)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    x_min, x_max, y_min, y_max = XY_ODE_type(ode_type)

    # plot traditional neural ODE
    for i in range(u_train_trad.shape[1]):
        ax1.plot(u_train_trad[:, i, 0], u_train_trad[:, i, 1], 'r', linewidth=1.0)
    ax1.plot([],[], 'r', linewidth=1.0, label='Training Dataset')
    for i in range(u_test_trad.shape[1]):
        ax1.plot(u_test_trad[:, i, 0], u_test_trad[:, i, 1], '#32cd32', linewidth=1.0)
    ax1.plot([],[], '#32cd32', linewidth=1.0, label='Testing Dataset')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Vanilla NODEs')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right')

    # plot semi-neural ODE
    for i in range(u_train_semi.shape[1]):
        ax2.plot(u_train_semi[:, i, 0], u_train_semi[:, i, 1], 'r', linewidth=1.0)
    ax2.plot([],[], 'r', linewidth=1.0, label='Training Dataset')
    for i in range(u_test_semi.shape[1]):
        ax2.plot(u_test_semi[:, i, 0], u_test_semi[:, i, 1], '#32cd32', linewidth=1.0)
    ax2.plot([],[], '#32cd32', linewidth=1.0, label='Testing Dataset')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('SA-NODEs')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.legend(loc='upper right')

    # plot exact solution
    for i in range(u_real.shape[1]):
        ax3.plot(u_real[:, i, 0], u_real[:, i, 1], 'b', linewidth=1.0)
    ax3.plot(0, 0, marker='o', markersize=1.0, color='b')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Exact')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    # save the figure
    plt.savefig('./figures/ODE_compare.png', bbox_inches='tight')
    plt.close()

def plot_ODEs_compare_error(t, error_trad, error_semi):
    t = t.detach().cpu().numpy()
    error_trad_mean = error_trad.mean(axis=1)
    error_semi_mean = error_semi.mean(axis=1)

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, error_trad_mean, 'r', linewidth=1.0)
    ax.plot(t, error_semi_mean, 'b', linewidth=1.0)

    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Errors of different NODEs')
    # legend
    ax.legend(['Vanilla NODEs', 'SA-NODEs'])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.savefig('./figures/ODE_compare_error.png', bbox_inches='tight')
    plt.close()

def plot_transport(t, anode, dt, transport_type):
    """
    Generate dataset and plot 2D transport equation solutions.
    Plots the initial condition (t=0), the SA-NODE solution, and the Exact solution
    at different time instances.
    """
    # Choose plotting parameters based on transport_type
    if transport_type == 'nonautonomous':
        l, cmap, vmin, vmax = 4, 'viridis', 0, 1.2
    elif transport_type == 'Doswell':
        l, cmap, vmin, vmax = 5, 'jet', -1, 1

    # Build the spatial grid
    N_x = 201
    x = np.linspace(-l, l, N_x)
    y = np.linspace(-l, l, N_x)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # Prepare figure and axes
    num_fig = 6
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(2, num_fig, figsize=((num_fig)*5, 10), dpi=300)
    axes = axes.flatten()

    # Plot t=0 initial condition for both panels
    with torch.no_grad():
        init_rho = init_cond(XY_tensor, transport_type)[:, 2].reshape(N_x, N_x).cpu().numpy()
    for idx, title in zip([0, num_fig], ['t=0 (SA-NODEs)', 't=0 (Exact)']):
        ax = axes[idx]
        im = ax.imshow(init_rho, extent=(-l, l, -l, l), origin='lower',
                       cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax.set(xlabel='x', ylabel='y', title=title)
    del init_rho, im
    torch.cuda.empty_cache(); gc.collect()

    # Determine which time indices to sample
    step = (t.shape[0]-1) // (num_fig-1)
    time_indices = list(range(step, t.shape[0], step))

    plot_idx = 1
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
            u0_fwd = init_cond(XY_mapped, transport_type)
            sol_fwd = batch_odeint(lambda y, τ: model(y, τ, transport_type),
                                   u0_fwd, t_fwd, batch_size=256)
            exact_img = sol_fwd[-1, :, 2].reshape(N_x, N_x).cpu().numpy()

        # Plot Exact
        ax_ex = axes[plot_idx + num_fig]
        im = ax_ex.imshow(exact_img, extent=(-l, l, -l, l), origin='lower',
                          cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax_ex.set(xlabel='x', ylabel='y', title=f't={int(round(time*dt))} (Exact)')
        del sol_bwd, sol_fwd, u_back, u0_bwd, exact_img
        torch.cuda.empty_cache(); gc.collect()

        # --- SA-NODE solution: backward then forward ---
        with torch.no_grad():
            sol_fwd_nn = anode.integrate(u0_fwd.to(anode.device), backward=0, eval_times=t_fwd.to(anode.device))
            nn_img = sol_fwd_nn[-1, :, 2].reshape(N_x, N_x).cpu().numpy()

        # Plot SA-NODE
        ax_nn = axes[plot_idx]
        im = ax_nn.imshow(nn_img, extent=(-l, l, -l, l), origin='lower',
                          cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax_nn.set(xlabel='x', ylabel='y', title=f't={int(round(time*dt))} (SA-NODEs)')
        plot_idx += 1

        # Cleanup
        del sol_fwd_nn, nn_img
        torch.cuda.empty_cache(); gc.collect()

    # Finalize layout and add a common colorbar
    plt.tight_layout()
    fig.colorbar(im, ax=axes, orientation='vertical',
                 fraction=0.02, pad=0.04, aspect=20)
    plt.savefig('./figures/Transport2D_combined.png', bbox_inches='tight')
    plt.close()
       
def model(u0, t, transport_type):
    x, y, z = u0
    if transport_type == 'nonautonomous':
        dydt = [(sin(x))/(1+t**2), (sin(y))/(1+t**2), -z*(cos(x)+cos(y))/(1+t**2)]
    elif transport_type == 'Doswell':
        r = sqrt(x**2+y**2)+1e-10
        dydt = [-y*(1./r)*2.58907*(sech(r)**2)*tanh(r), x*(1./r)*2.58907*(sech(r)**2)*tanh(r),0]
    return dydt

def init_cond(XY, transport_type):
    if transport_type == 'nonautonomous':
        sigma = 2
        u_init = torch.exp(-torch.sum(XY**2, dim=1)/(sigma**2)).reshape(-1, 1)  
    elif transport_type == 'Doswell':
        u_init = torch.tanh(XY[:, 1]/0.1).reshape(-1, 1)
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


def plot_transport_compare(t, node, sanode, dt, transport_type):
    """
    Generate dataset and plot 2D transport equation solutions for comparison.
    This function plots the initial condition and the solution at various time steps
    for SA-NODEs, Vanilla NODEs, and the exact solution.
    """
    # Choose plotting parameters based on transport_type
    if transport_type == 'nonautonomous':
        l, cmap, vmin, vmax = 4, 'viridis', 0, 1.2
    elif transport_type == 'Doswell':
        l, cmap, vmin, vmax = 5, 'jet', -1, 1

    # Build the spatial grid
    N_x = 201
    x = np.linspace(-l, l, N_x)
    y = np.linspace(-l, l, N_x)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=-1)
    XY_tensor = torch.tensor(XY, dtype=torch.float32)

    # Prepare figure and axes
    num_fig = 6
    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(3, num_fig, figsize=((num_fig)*5, 15), dpi=300)
    axes = axes.flatten()

    # Plot t=0 initial condition for SA-NODEs, Vanilla NODEs, and Exact solution
    with torch.no_grad():
        rho0 = init_cond(XY_tensor, transport_type)[:, 2].reshape(N_x, N_x).cpu().numpy()
    for idx, title in zip([0, num_fig, 2 * num_fig],
                          ['t=0 (SA-NODEs)', 't=0 (Vanilla NODEs)', 't=0 (Exact)']):
        ax = axes[idx]
        im = ax.imshow(rho0, extent=(-l, l, -l, l), origin='lower',
                       cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax.set(xlabel='x', ylabel='y', title=title)
    del rho0, im
    torch.cuda.empty_cache(); gc.collect()

    # Determine which time indices to sample
    step = (t.shape[0] - 1) // (num_fig - 1)
    time_indices = list(range(step, t.shape[0], step))

    plot_idx = 1
    for time in time_indices:
        # Build time arrays for backward and forward integration
        t_fwd = torch.linspace(0., time * dt, time + 1)
        t_bwd = torch.linspace(time * dt, 0., time + 1)

        # ------------------------
        # Exact solution: backward then forward
        # ------------------------
        with torch.no_grad():
            # Backward integration on the true model (CPU)
            u0_bwd = torch.cat([XY_tensor, torch.zeros(XY_tensor.shape[0], 1)], dim=1).cpu()
            sol_bwd = batch_odeint(lambda y, τ: model(y, τ, transport_type),
                                   u0_bwd, t_bwd, batch_size=256)
            u_back = sol_bwd[-1]               # shape [N, 3]

            # Forward integration from mapped points (CPU)
            XY_mapped = u_back[:, :2]
            u0_fwd = init_cond(XY_mapped, transport_type)
            sol_fwd = batch_odeint(lambda y, τ: model(y, τ, transport_type),
                                   u0_fwd, t_fwd, batch_size=256)
            exact_img = sol_fwd[-1, :, 2].reshape(N_x, N_x).cpu().numpy()

        # Plot Exact
        ax_ex = axes[plot_idx + 2*num_fig]
        im = ax_ex.imshow(exact_img, extent=(-l, l, -l, l), origin='lower',
                          cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax_ex.set(xlabel='x', ylabel='y',
                  title=f't={int(round(time * dt))} (Exact)')
        del sol_bwd, sol_fwd, u_back, u0_bwd, exact_img
        torch.cuda.empty_cache(); gc.collect()

        # ------------------------
        # Vanilla NODEs prediction
        # ------------------------
        with torch.no_grad():
            sol_fwd_nn = node.integrate(u0_fwd.to(node.device), backward=0, eval_times=t_fwd.to(node.device))
            node_img = sol_fwd_nn[-1, :, 2].reshape(N_x, N_x).cpu().numpy()

        ax_node = axes[plot_idx + num_fig]
        im = ax_node.imshow(node_img, extent=(-l, l, -l, l), origin='lower',
                            cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax_node.set(xlabel='x', ylabel='y',
                    title=f't={int(round(time * dt))} (Vanilla NODEs)')
        del node_img, sol_fwd_nn
        torch.cuda.empty_cache(); gc.collect()

        # ------------------------
        # SA-NODE solution: backward then forward
        # ------------------------
        with torch.no_grad():
            sol_fwd_nn = sanode.integrate(u0_fwd.to(sanode.device), backward=0, eval_times=t_fwd.to(sanode.device))
            nn_img = sol_fwd_nn[-1, :, 2].reshape(N_x, N_x).cpu().numpy()

        ax_nn = axes[plot_idx]
        im = ax_nn.imshow(nn_img, extent=(-l, l, -l, l), origin='lower',
                          cmap=cmap, interpolation='bicubic', vmin=vmin, vmax=vmax)
        ax_nn.set(xlabel='x', ylabel='y',
                  title=f't={int(round(time * dt))} (SA-NODEs)')
        plot_idx += 1

        del sol_fwd_nn, nn_img, u0_fwd
        torch.cuda.empty_cache(); gc.collect()

    # Finalize layout and add a common colorbar
    plt.tight_layout()
    fig.colorbar(im, ax=axes, orientation='vertical',
                 fraction=0.02, pad=0.04, aspect=30)
    plt.savefig('./figures/Transport2D_compare.png', bbox_inches='tight')
    plt.close()


def plot_transport_error(t, error_list_train, error_list_test):
    # plot the errors in error_list_train and error_list_test at different t in one figure with different colors
    # t = t.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, error_list_train, 'r', linewidth=1.0, label='Training Dataset')
    ax.plot(t, error_list_test, 'b', linewidth=1.0, label='Testing Dataset')

    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Errors of SA-NODEs')

    ax.legend(loc='upper left')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.savefig('./figures/Transport2D_error.png', bbox_inches='tight')
    plt.close()

def plot_transport_compare_error(t, error_node, error_sanode):
    # plot the errors in error_list_train and error_list_test at different t in one figure with different colors
    # t = t.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(t, error_node, 'r', linewidth=1.0)
    ax.plot(t, error_sanode, 'b', linewidth=1.0)

    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.set_title('Errors of different NODEs')

    ax.legend(['Vanilla NODEs', 'SA-NODEs'], loc='upper left')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.savefig('./figures/Transport2D_compare_error.png', bbox_inches='tight')
    plt.close()

def XY_ODE_type(ode_type):
    if ode_type == 'nonlinear_nonautonomous':
        return -3, 3, -4, 4
    elif ode_type == 'nonlinear_autonomous':
        return -8, 8, -8, 8
    elif ode_type == 'linear_autonomous':
        return -3, 3, -3, 3
    elif ode_type == 'linear_nonautonomous':
        return -4, 10, -4, 10
    else:
        raise ValueError('Invalid ODE type')

def plot_epoch_error(epoch_list, sanode_error_list, node_error_list):
    # plot the errors in error_list_train and error_list_test at different t in one figure with different colors

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(epoch_list, sanode_error_list, 'r', linewidth=1.0)
    ax.plot(epoch_list, node_error_list, 'b', linewidth=1.0)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Comparison of errors by epochs')

    ax.legend(['SA-NODEs', 'Vanilla NODEs'], loc='upper right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.savefig('./figures/epoch_error.png', bbox_inches='tight')
    plt.close()

def plot_data_error(data_list, sanode_error_list, node_error_list):
    # plot the errors in error_list_train and error_list_test at different t in one figure with different colors

    fig = plt.figure(figsize=(6, 5),dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(data_list, sanode_error_list, 'r', linewidth=1.0)
    ax.plot(data_list, node_error_list, 'b', linewidth=1.0)

    ax.set_xlabel('Trajectories')
    ax.set_ylabel('Error')
    ax.set_title('Comparison of errors by training size ')

    ax.legend(['SA-NODEs', 'Vanilla NODEs'], loc='upper right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.savefig('./figures/data_error.png', bbox_inches='tight')
    plt.close()