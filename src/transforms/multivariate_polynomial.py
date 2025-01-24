import torch
import torch.nn as nn
import itertools
import math

from nflows.transforms import Transform

class MultivariatePolynomialTransform(Transform):
    def __init__(self, features, partitioning, order):
        assert sum(partitioning) == features
        super().__init__()
        self.features = features
        self.partitioning = partitioning
        self.num_partitions = len(partitioning)
        self.order = order

        # Calculate the number of polynomial terms for each partition
        self.num_terms = [self._calculate_num_terms(p-1, order) for p in partitioning]

        # Create learnable parameters for each partition
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.zeros(num_terms))
            for num_terms in self.num_terms
        ])

    def _calculate_num_terms(self, num_features, order):
        return sum(self._combinations_with_replacement(num_features, i) for i in range(1, order + 1))

    def _combinations_with_replacement(self, n, r):
        return math.comb(n + r - 1, r)

    def forward(self, x, context=None):
        z = x.clone()
        log_abs_det = torch.zeros(1, device=x.device)
        for i in range(self.num_partitions):
            z_i = x[:, sum(self.partitioning[:i])+1:sum(self.partitioning[:i+1])]
            z[:,sum(self.partitioning[:i])] += (self.coefficients[i][None] * self.multivariate_polynomial(z_i, self.order)).sum(1)
        return z, log_abs_det
    
    def inverse(self, z, context=None):
        x = z.clone()
        log_abs_det = torch.zeros(1, device=z.device)
        for i in range(self.num_partitions):
            x_i = z[:, sum(self.partitioning[:i])+1:sum(self.partitioning[:i+1])]
            x[:,sum(self.partitioning[:i])] -= (self.coefficients[i][None] * self.multivariate_polynomial(x_i, self.order)).sum(1)
        return x, log_abs_det

    def multivariate_polynomial(self, x, order):
        assert 1 <= order <= 3, "Order must be 1, 2, or 3"
        
        _, n_features = x.shape
        result = [x]  # Start with first-order terms

        if order >= 2:
            # Add second-order terms without duplicates
            second_order = []
            for i, j in itertools.combinations_with_replacement(range(n_features), 2):
                second_order.append(x[:, i] * x[:, j])
            result.append(torch.stack(second_order, dim=1))

        if order == 3:
            # Add third-order terms without duplicates
            third_order = []
            for i, j, k in itertools.combinations_with_replacement(range(n_features), 3):
                third_order.append(x[:, i] * x[:, j] * x[:, k])
            result.append(torch.stack(third_order, dim=1))

        # Concatenate all terms
        return torch.cat(result, dim=1)