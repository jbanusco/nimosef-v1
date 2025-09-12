import torch
import torch.nn as nn
from typing import Tuple


class WIRE(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_size: Input features
            out_size; Output features
            bias: if True, enable bias for the linear operation
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_size, out_size, bias=True, is_first=False,
                 omega_0=10, sigma_0=10, trainable=False):
        super(WIRE, self).__init__()        

        self.omega_0 = nn.Parameter(omega_0*torch.ones(1), requires_grad=trainable)
        self.scale_0 = nn.Parameter(sigma_0*torch.ones(1), requires_grad=trainable)
        self.is_first = is_first

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        self.freqs = nn.Linear(in_size, out_size, bias=bias, dtype=dtype)
        self.scale = nn.Linear(in_size, out_size, bias=bias, dtype=dtype)

    def forward(self, x):                    
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))

        return x
    


class INR(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, hidden_size: int, out_size: int, 
                 num_hidden_layers: int, use_residual: bool = True):
        super(INR, self).__init__()

        in_fts = coord_size + embed_size
        # hidden_fts = int(hidden_size / 2)  # Complex numbers
        hidden_fts = hidden_size  # Complex numbers
        # dtype = torch.cfloat
        dtype = torch.float

        self.net = nn.ModuleList()

        # Append first layer
        # self.net.append(WIRE(in_fts, hidden_fts, is_first=True, trainable=True, omega_0=10, sigma_0=10))
        self.net.append(WIRE(in_fts, hidden_fts, is_first=True, trainable=False, omega_0=10, sigma_0=10))

        # Append layers
        for _ in range(num_hidden_layers - 1):
            # self.net.append(WIRE(hidden_fts, hidden_fts, is_first=True, trainable=True, omega_0=5, sigma_0=5))
            self.net.append(WIRE(hidden_fts, hidden_fts, is_first=True, trainable=False, omega_0=10, sigma_0=10))

        # Append last layer
        final_layer = nn.Linear(hidden_fts, out_size, dtype=dtype)
        self.net.append(final_layer)

        self.use_residual = use_residual
        if not self.use_residual:
            self.net = nn.Sequential(*self.net)
        

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        _, x_prev = x
        x = torch.cat(x, dim=1)
        if self.use_residual:
            for layer in self.net:
                x = layer(x) + x_prev
                x_prev = x
        else:
            x = self.net(x).real
        
        return x
    
