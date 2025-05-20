import torch
import torch.nn as nn


class ReLUk(nn.Module):
    def __init__(self):
        super(ReLUk, self).__init__()
        self.k = 2
    def forward(self, x):
        return torch.pow(torch.relu(x), self.k)
    
def activations(name):
    """
    Return the activation function based on the name.
    """
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'leakyrelu':
        return nn.LeakyReLU()
    elif name == 'reluk':
        return ReLUk()
    elif name == 'sin':
        return lambda x: torch.sin(x)
    elif name == 'cos':
        return lambda x: torch.cos(x)

def dactivations(name):
    """
    Return the derivative of the activation function based on the name.
    """
    if name == 'relu':
        return dReLU
    elif name == 'reluk':
        return dReLUk
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
    elif name == 'sin':
        return lambda x: torch.cos(x)
    elif name == 'cos':
        return lambda x: -torch.sin(x)
    elif name == 'tanh':
        return lambda x: 1 - torch.tanh(x).pow(2)

    
def dReLUk(x):
    k = 2
    # derivative of ReLU
    return k*torch.pow(torch.relu(x), k-1)

def dReLU(x):
    # derivative of ReLU
    return (x >= 0).float()

