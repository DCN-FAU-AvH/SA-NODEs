import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from models.acti_func import activations, dactivations
import numpy as np

class NeuralODE(nn.Module):
    """
    A NeuralODE model that integrates dynamics, ODE integration, and flow mapping.

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
        super(NeuralODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.input_dim = data_dim
        self.output_dim = output_dim
        self.adjoint = adjoint
        self.T = T
        self.time_steps = time_steps
        self.non_linearity = activations(non_linearity)

        ##-- R^{d_aug} -> R^{d_hid} layer -- 
        blocks1 = [nn.Linear(self.input_dim, hidden_dim) for _ in range(self.time_steps)]
        self.fc1_time = nn.Sequential(*blocks1) #this does not represent multiple layers with non-linearities but the different discrete points in time of the weight and bias function.
        ##-- R^{d_hid} -> R^{d_aug} layer --
        blocks3 = [nn.Linear(hidden_dim, self.input_dim, bias=False) for _ in range(self.time_steps)]
        self.fc3_time = nn.Sequential(*blocks3)

        # Initialize Xavier weights and zero biases 
        for block1 in self.fc1_time:
            nn.init.xavier_uniform_(block1.weight)
            nn.init.zeros_(block1.bias)
        for block3 in self.fc3_time:
            nn.init.xavier_uniform_(block3.weight)

    def dynamics(self, t, x):
        """
        Defines the dynamics function f(x(t), t) for the Neural ODE.

        Args:
            t (float): Current time.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the dynamics function.
        """
        dt = self.T/(self.time_steps-1)
        k = int(t/dt)
                                                       
        w1_t = self.fc1_time[k].weight
        b1_t = self.fc1_time[k].bias
        w2_t = self.fc3_time[k].weight
        out = self.non_linearity(x.matmul(w1_t.t()) + b1_t)
        out = out.matmul(w2_t.t())

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

class NeuralODE_transport(nn.Module):
    """
    A NeuralODE model that integrates dynamics, ODE integration, and flow mapping.

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
        super(NeuralODE_transport, self).__init__()
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

        ##-- R^{d_aug} -> R^{d_hid} layer -- 
        blocks1 = [nn.Linear(self.input_dim, hidden_dim) for _ in range(self.time_steps)]
        self.fc1_time = nn.Sequential(*blocks1) #this does not represent multiple layers with non-linearities but the different discrete points in time of the weight and bias function.
        ##-- R^{d_hid} -> R^{d_aug} layer --
        blocks3 = [nn.Linear(hidden_dim, self.input_dim, bias=False) for _ in range(self.time_steps)]
        self.fc3_time = nn.Sequential(*blocks3)

        # Initialize Xavier weights and zero biases 
        for block1 in self.fc1_time:
            nn.init.xavier_uniform_(block1.weight)
            nn.init.zeros_(block1.bias)
        for block3 in self.fc3_time:
            nn.init.xavier_uniform_(block3.weight)

    def dynamics(self, t, x):                                    
        """
        Defines the dynamics function f(x(t), t) for the Neural ODE.

        Args:
            t (float): Current time.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the dynamics function.
        """
        dt = self.T/(self.time_steps-1)
        k = int(t/dt)
        # split x into X and Rho with gradient tracking
        X = x[:, 0:2].requires_grad_(True).to(self.device)
        Rho = x[:, 2].reshape(-1, 1).requires_grad_(True).to(self.device)

        w1_t = self.fc1_time[k].weight
        b1_t = self.fc1_time[k].bias
        w2_t = self.fc3_time[k].weight
        out = self.non_linearity(X.matmul(w1_t.t()) + b1_t)
        out_x = out.matmul(w2_t.t())

        # Compute rho'(t) = f'(x(t), u(t))*rho(t)          
        out_rho = self.dactivate(X.matmul(w1_t.t()) + b1_t).matmul((w1_t*w2_t.t()).sum(dim=1).reshape(-1, 1))
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
