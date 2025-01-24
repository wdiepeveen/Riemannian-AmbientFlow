import torch
import torch.nn as nn

from nflows.transforms import Transform

class LinearTransform(Transform):
    def __init__(self, features, efficient_inverse=False):
        super().__init__()
        self.features = features
        self.lower = nn.Parameter(torch.tril(torch.eye(features, features), diagonal=-1))
        self.upper = nn.Parameter(torch.triu(torch.eye(features, features)))
        self.bias = nn.Parameter(torch.zeros(features))
        self.efficient_inverse = efficient_inverse

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

    def _forward(self, x, context=None): # TODO make this more efficient
        weight = torch.matmul(torch.tril(self.lower, diagonal=-1) + torch.eye(self.features), torch.triu(self.upper))
        z = torch.matmul(x, weight.t()) + self.bias
        log_abs_det = torch.sum(torch.log(torch.abs(torch.diag(self.upper))))
        return z, log_abs_det.expand(x.shape[0])

    def _inverse(self, z, context=None): # TODO make this more efficient
        weight = torch.matmul(torch.tril(self.lower, diagonal=-1) + torch.eye(self.features), torch.triu(self.upper))
        x = torch.matmul(z - self.bias, torch.inverse(weight).t())
        log_abs_det = -torch.sum(torch.log(torch.abs(torch.diag(self.upper))))
        return x, log_abs_det.expand(z.shape[0])