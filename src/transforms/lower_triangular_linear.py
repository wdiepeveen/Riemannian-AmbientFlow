import torch
import torch.nn as nn

from nflows.transforms import Transform

class LowerTriangularLinearTransform(Transform):
    def __init__(self, features, bias=True, unit_det=True, efficient_inverse=False):
        super().__init__()
        self.features = features
        self.weight = nn.Parameter(torch.tril(torch.eye(features, features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.bias = torch.zeros(features)
        
        self.efficient_inverse = efficient_inverse
        self.unit_det = unit_det

    def forward(self, x, context=None):
        if self.efficient_inverse:
            return self._inverse(x, context=context)
        else:
            return self._forward(x, context=context)
        
    def inverse(self, z, context=None):
        if self.efficient_inverse:
            return self._forward(z, context=context)
        else:
            return self._inverse(z, context=context)

    def _forward(self, x, context=None): 
        if self.unit_det:
            z = x @ (torch.tril(self.weight.to(x.device), diagonal=-1) + torch.eye(self.features).to(x.device)).t()
            log_abs_det = torch.zeros(1, device=x.device)
        else:
            z = x @ (torch.tril(self.weight).to(x.device)).t()
            log_abs_det = torch.sum(torch.log(torch.abs(torch.diag(self.weight.to(x.device)))))
        z += self.bias.to(x.device)
        return z, log_abs_det.expand(x.shape[0])

    def _inverse(self, z, context=None):
        if self.unit_det:
            x = torch.matmul(z - self.bias.to(z.device), torch.inverse(torch.tril(self.weight.to(z.device), diagonal=-1) + torch.eye(self.features).to(z.device)).t())
            log_abs_det = torch.zeros(1, device=z.device)
        else:
            x = torch.matmul(z - self.bias.to(z.device), torch.inverse(torch.tril(self.weight).to(z.device)).t())
            log_abs_det = -torch.sum(torch.log(torch.abs(torch.diag(self.weight)))).to(z.device)
        return x, log_abs_det.expand(z.shape[0])