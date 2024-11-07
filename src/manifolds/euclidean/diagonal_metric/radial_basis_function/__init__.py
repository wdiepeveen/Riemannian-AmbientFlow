import torch

from src.manifolds.euclidean.diagonal_metric import DiagonalMetricEuclidean

class RadialBasisFunctionEuclidean(DiagonalMetricEuclidean):
    """ Base class describing Euclidean space of dimension d under a radial basis function (RBF) metric """

    def __init__(self, d, rho): 
        super().__init__(d)
        self.rho = rho

    def radial_basis_function(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def metric_tensor_diagonal(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        return 1 / (self.radial_basis_function(x)[:,None] * torch.ones(self.d)[None] + self.rho)