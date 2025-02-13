import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math

from nflows.transforms import Transform

from src.nn.module.conv2d.masked_conv import MaskedConv2d
from src.nn.module.polynomial_activation import PolynomialActivation

class PolynomialParityTransform(Transform):
    def __init__(self, features, context_features=None, order=2, parity=0):
        assert 1 <= order <= 3, "Order must be 1, 2, or 3"
        super().__init__()
        self.d = features

        self.order = order

        self.parity = parity % 2
        self.data_mask = self.generate_data_mask()

        self.poly = PolynomialActivation(self.d, order=self.order)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,  # Symmetric padding for kernel_size=3
            bias=False   # No bias term needed
        )
        if context_features is not None:
             self.lin = nn.Linear(context_features, features)

        # Manually set filter weights to [1, 0, 1]
        with torch.no_grad():
            self.conv.weight.data = torch.tensor(
                [[[1.0, 0.0, 1.0]]],  # Shape: (out_channels, in_channels, kernel_size)
                dtype=torch.float32
            )
            self.conv.weight.requires_grad_(False)  # Freeze weights

    def forward(self, x, context=None):
        x_ = x.clone()
        log_abs_det = torch.zeros(1, device=x.device)
        
        # Convolve
        c = self.conv(self.poly(x_).unsqueeze(1)).squeeze(1) 
        
        # Apply non-linearity
        z = torch.zeros_like(x_)
        z[:,~self.data_mask.bool()] = x_[:,~self.data_mask.bool()]
        z[:,self.data_mask.bool()] = x_[:,self.data_mask.bool()] + c[:,self.data_mask.bool()]

        return z, log_abs_det.expand(x.shape[0])
    
    def inverse(self, z, context=None):
        z_ = z.clone()
        log_abs_det = torch.zeros(1, device=z.device)
        
        # Convolve
        c = self.conv(self.poly(z_).unsqueeze(1)).squeeze(1)
        
        # Apply non-linearity
        x = torch.zeros_like(z_)
        x[:,~self.data_mask.bool()] = z_[:,~self.data_mask.bool()]
        x[:,self.data_mask.bool()] = z_[:,self.data_mask.bool()] - c[:,self.data_mask.bool()]

        return x, log_abs_det.expand(z.shape[0])
    
    def generate_data_mask(self):
        mask = torch.zeros(self.d)
        for i in range(self.d):
                mask[i] = 1 if i % 2 == self.parity else 0
        return mask
        