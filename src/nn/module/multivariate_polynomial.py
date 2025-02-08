import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

class MultivariatePolynomial(nn.Module):
    def __init__(self, d_in, partitioning, order):
        super().__init__()
        assert sum(partitioning) == d_in
        self.partitioning = partitioning
        self.num_partitions = len(partitioning)
        self.order = order

        # Calculate the number of polynomial terms for each partition
        self.num_terms = [self._calculate_num_terms(p, order) for p in partitioning]

        # Create learnable parameters for each partition
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.randn(num_terms))
            for num_terms in self.num_terms
        ])

        # Linear transformation to map output to two numbers (scale and shift)
        self.linear = nn.Linear(self.num_partitions, 2)

    def _calculate_num_terms(self, num_features, order):
        return sum(self._combinations_with_replacement(num_features, i) for i in range(1, order + 1))

    def _combinations_with_replacement(self, n, r):
        return math.comb(n + r - 1, r)
    
    def forward(self, x, context=None):
        z = torch.zeros(x.shape[0], self.num_partitions, device=x.device)
        for i in range(self.num_partitions):
            z_i = x[:, sum(self.partitioning[:i]):sum(self.partitioning[:i+1])]
            z[:, i] = (self.coefficients[i][None] * self.multivariate_polynomial(z_i, self.order)).sum(1)
        return self.linear(z)  # Returns two values per input
    
    def multivariate_polynomial(self, x, order):
        assert 1 <= order <= 3, "Order must be 1, 2, or 3"
        
        batch_size, n_features = x.shape
        result = [x]  # Start with first-order terms

        if order >= 2:
            # Add second-order terms
            second_order = []
            for i, j in itertools.combinations_with_replacement(range(n_features), 2):
                second_order.append(x[:, i] * x[:, j])
            result.append(torch.stack(second_order, dim=1))

        if order == 3:
            # Add third-order terms
            third_order = []
            for i, j, k in itertools.combinations_with_replacement(range(n_features), 3):
                third_order.append(x[:, i] * x[:, j] * x[:, k])
            result.append(torch.stack(third_order, dim=1))

        # Concatenate all terms
        return torch.cat(result, dim=1)