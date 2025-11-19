import torch
import torch.nn as nn

from src.nn.module.activation import Activation

class TanhActivation(Activation):
    def __init__(self, num_features, order=2):
        super().__init__(num_features)
        self.order = order
        self.coefficients = nn.Parameter(torch.zeros(self.d, self.order+1))
        
    def forward(self, x):
        x_tanh = torch.tanh(x)
        powers_list = [torch.ones_like(x_tanh)]  # Start with power 0

        if x.dim() == 2:  # Vector data (batch, features)
            for k in range(1, self.order + 1):
                powers_list.append(powers_list[-1] * x_tanh)

            # Stack along the last dimension to get shape (batch, features, order+1)
            x_powers = torch.stack(powers_list, dim=-1)
            return torch.sum(self.coefficients * x_powers, dim=-1)
        elif x.dim() == 4:  # Image data (batch, channels, height, width)
            assert x.shape[1] == self.d
            for k in range(1, self.order + 1):
                powers_list.append(powers_list[-1] * x_tanh)

            # Stack along a new last dimension to get shape (batch, channels, height, width, order+1)
            x_powers = torch.stack(powers_list, dim=-1)
            coefficients = self.coefficients.view(self.d, 1, 1, self.order+1)
            return torch.sum(coefficients * x_powers, dim=-1)
        else:
            raise ValueError("Input must be 2D (vector) or 4D (image)")