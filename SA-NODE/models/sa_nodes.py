import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from models.acti_func import activations, dactivations
import numpy as np

class SANODE(nn.Module):
    """
    A SA-NODE model that integrates dynamics, ODE integration, and flow mapping.

    Attributes:
        device (torch.device): Computation device.
        data_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the final output.
        tol (float): Tolerance for the ODE solver.
        adjoint (bool): Whether to use the adjoint method for gradients.
        T (float): Total integration time.
        time_steps (int): Number of time steps.
    """

    def __init__(self, device, data_dim, hidden_dim, output_dim=2,
                 non_linearity='tanh', adjoint=False, T=10, time_steps=10):
        super(SANODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.input_dim = data_dim
        self.output_dim = output_dim
        self.adjoint = adjoint
        self.T = T
        self.time_steps = time_steps
        self.non_linearity = activations(non_linearity)

        # Define dynamics layers for two branches
        self.fc1_time_1 = nn.Linear(self.input_dim, hidden_dim, bias=False).to(device)
        self.fc1_time_2 = nn.Linear(self.input_dim, hidden_dim, bias=False).to(device)
        self.b_time_1 = nn.Linear(1, hidden_dim).to(device)
        self.b_time_2 = nn.Linear(1, hidden_dim).to(device)
        self.fc3_time_1 = nn.Linear(hidden_dim, 1, bias=False).to(device)
        self.fc3_time_2 = nn.Linear(hidden_dim, 1, bias=False).to(device)

        # Optional projection layer for the final output
        # self.linear_layer = nn.Linear(self.input_dim, output_dim).to(device)

        # Xavier initialization for weights and zero initialization for biases
        nn.init.xavier_normal_(self.fc1_time_1.weight)
        nn.init.xavier_normal_(self.fc1_time_2.weight)
        nn.init.xavier_normal_(self.fc3_time_1.weight)
        nn.init.xavier_normal_(self.fc3_time_2.weight)
        nn.init.xavier_normal_(self.b_time_1.weight)
        nn.init.xavier_normal_(self.b_time_2.weight)
        nn.init.zeros_(self.b_time_1.bias)
        nn.init.zeros_(self.b_time_2.bias)

    def dynamics(self, t, x):
        """
        Defines the dynamics function f(x(t), t) for the Neural ODE.

        Args:
            t (float): Current time.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the dynamics function.
        """
        # Convert time to tensor and move to the specified device
        t_tensor = torch.tensor([t]).float().to(self.device)
        b_t_1 = self.b_time_1(t_tensor)
        b_t_2 = self.b_time_2(t_tensor)

        # Compute branch outputs
        out_1 = self.non_linearity(x.matmul(self.fc1_time_1.weight.t()) + b_t_1)
        out_2 = self.non_linearity(x.matmul(self.fc1_time_2.weight.t()) + b_t_2)
        out_1 = out_1.matmul(self.fc3_time_1.weight.t())
        out_2 = out_2.matmul(self.fc3_time_2.weight.t())

        # Concatenate outputs from both branches
        out = torch.cat((out_1, out_2), dim=1)
        return out

    def integrate(self, u, eval_times=None):
        """
        Integrates the ODE for the given initial condition u.

        Args:
            u (torch.Tensor): Initial condition.
            eval_times (torch.Tensor, optional): Specific time points for evaluation.
                Defaults to None, which returns only the final state.

        Returns:
            torch.Tensor: The ODE solution evaluated at the specified times.
        """
        dt = self.T / (self.time_steps - 1)
        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(u)
        else:
            integration_time = eval_times.type_as(u)

        if self.adjoint:
            out = odeint_adjoint(self.dynamics, u, integration_time, method='euler', options={'step_size': dt})
        else:
            out = odeint(self.dynamics, u, integration_time, method='euler', options={'step_size': dt})
            
        return out[1] if eval_times is None else out[1:]
    
    def trajectory(self, u, timesteps):
        """
        Computes the trajectory of the ODE solution at evenly spaced time points.

        Args:
            u (torch.Tensor): Initial condition.
            timesteps (int): Number of time steps in the trajectory.

        Returns:
            torch.Tensor: The full trajectory of the ODE solution.
        """
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.integrate(u, eval_times=integration_time)
    
    def forward(self, u):
        """
        Forward pass computes the final state and the full trajectory of the ODE.

        Args:
            u (torch.Tensor): Initial condition.

        Returns:
            tuple: (features, traj)
                features: ODE solution at the final time,
                traj: Full trajectory of the ODE solution.
        """
        features = self.integrate(u)
        self.traj = self.trajectory(u, self.time_steps)
        # Uncomment the following line to apply a linear projection to the final features:
        # features = self.linear_layer(features)
        return features, self.traj


class SANODE_transport(nn.Module):
    """
    A SA-NODE model that integrates dynamics, ODE integration, and flow mapping.

    Attributes:
        device (torch.device): Computation device.
        data_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the final output.
        tol (float): Tolerance for the ODE solver.
        adjoint (bool): Whether to use the adjoint method for gradients.
        T (float): Total integration time.
        time_steps (int): Number of time steps.
    """

    def __init__(self, device, data_dim, hidden_dim, output_dim=2,
                 non_linearity='tanh', adjoint=False, T=10, time_steps=10):
        super(SANODE_transport, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.input_dim = data_dim
        self.output_dim = output_dim
        self.adjoint = adjoint
        self.T = T
        self.time_steps = time_steps
        self.non_linearity = activations(non_linearity)
        self.dactivate = dactivations(non_linearity)


        # Define dynamics layers for two branches
        self.fc1_time_1 = nn.Linear(self.input_dim, hidden_dim, bias=False).to(device)
        self.fc1_time_2 = nn.Linear(self.input_dim, hidden_dim, bias=False).to(device)
        self.b_time_1 = nn.Linear(1, hidden_dim).to(device)
        self.b_time_2 = nn.Linear(1, hidden_dim).to(device)
        self.fc3_time_1 = nn.Linear(hidden_dim, 1, bias=False).to(device)
        self.fc3_time_2 = nn.Linear(hidden_dim, 1, bias=False).to(device)

        # Optional projection layer for the final output
        # self.linear_layer = nn.Linear(self.input_dim, output_dim).to(device)

        # Xavier initialization for weights and zero initialization for biases
        nn.init.xavier_normal_(self.fc1_time_1.weight)
        nn.init.xavier_normal_(self.fc1_time_2.weight)
        nn.init.xavier_normal_(self.fc3_time_1.weight)
        nn.init.xavier_normal_(self.fc3_time_2.weight)
        nn.init.xavier_normal_(self.b_time_1.weight)
        nn.init.xavier_normal_(self.b_time_2.weight)
        nn.init.zeros_(self.b_time_1.bias)
        nn.init.zeros_(self.b_time_2.bias)

    def dynamics(self, t, x):
        """
        Defines the dynamics function f(x(t), t) for the Neural ODE.

        Args:
            t (float): Current time.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the dynamics function.
        """
        # split x into X and Rho with gradient tracking
        X = x[:, 0:2].requires_grad_(True).to(self.device)
        Rho = x[:, 2].reshape(-1, 1).requires_grad_(True).to(self.device)

        w1_t_1 = self.fc1_time_1.weight
        w1_t_2 = self.fc1_time_2.weight
        #b_t = self.fc2_time[k].bias
        t = torch.tensor([t]).float().to(self.device)
        b_t_1 = self.b_time_1(t)
        b_t_2 = self.b_time_2(t)
        w2_t_1 = self.fc3_time_1.weight
        w2_t_2 = self.fc3_time_2.weight

        out_1 = self.non_linearity(X.matmul(w1_t_1.t()) + b_t_1)
        out_2 = self.non_linearity(X.matmul(w1_t_2.t()) + b_t_2)
        out_1 = out_1.matmul(w2_t_1.t())
        out_2 = out_2.matmul(w2_t_2.t())
        out_x = torch.cat((out_1, out_2), 1)

        # Compute rho'(t) = f'(x(t), u(t))*rho(t)          
        out_rho = self.dactivate(X.matmul(w1_t_1.t()) + b_t_1).matmul((w1_t_1*w2_t_1.t()).sum(dim=1).reshape(-1, 1))+\
                    self.dactivate(X.matmul(w1_t_2.t()) + b_t_2).matmul((w1_t_2*w2_t_2.t()).sum(dim=1).reshape(-1, 1))
        out_rho = -1*out_rho*Rho


        out = torch.cat((out_x, out_rho), dim=1)

        return out

    def integrate(self, u, backward, eval_times=None):
        """
        Integrates the ODE for the given initial condition u.

        Args:
            u (torch.Tensor): Initial condition.
            eval_times (torch.Tensor, optional): Specific time points for evaluation.
                Defaults to None, which returns only the final state.

        Returns:
            torch.Tensor: The ODE solution evaluated at the specified times.
        """
        dt = self.T/(self.time_steps-1)

        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(u)
            if backward == 1:
                integration_time = np.flipud(integration_time.cpu()).copy()
                integration_time = torch.tensor(integration_time, dtype=torch.float32)
        else:
            integration_time = eval_times.type_as(u)
            if backward == 1:
                integration_time = np.flipud(integration_time.cpu()).copy()
                integration_time = torch.tensor(integration_time, dtype=torch.float32)
                # Delete the first element
                integration_time = integration_time[1:]

        out = odeint(self.dynamics, u, 
                        integration_time, 
                        method='euler', 
                        options={'step_size': dt})
        
        if eval_times is None:
            return out[1]
        else:
            return out[1:]
    
    def trajectory(self, u, timesteps, backward):
        """
        Computes the trajectory of the ODE solution at evenly spaced time points.

        Args:
            u (torch.Tensor): Initial condition.
            timesteps (int): Number of time steps in the trajectory.

        Returns:
            torch.Tensor: The full trajectory of the ODE solution.
        """
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.integrate(u, backward, eval_times=integration_time)
    
    def forward(self, u, backward=0):
        """
        Forward pass computes the final state and the full trajectory of the ODE.

        Args:
            u (torch.Tensor): Initial condition.

        Returns:
            tuple: (features, traj)
                features: ODE solution at the final time,
                traj: Full trajectory of the ODE solution.
        """
        features = self.integrate(u, backward)
        self.traj = self.trajectory(u, self.time_steps, backward)
        # Uncomment the following line to apply a linear projection to the final features:
        # features = self.linear_layer(features)
        return features, self.traj


