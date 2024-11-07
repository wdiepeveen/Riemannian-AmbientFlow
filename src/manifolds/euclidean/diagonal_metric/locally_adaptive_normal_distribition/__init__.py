import torch

from src.manifolds.euclidean.diagonal_metric import DiagonalMetricEuclidean

class LocallyAdaptiveNormalDistributionEuclidean(DiagonalMetricEuclidean):
    """ Base class describing Euclidean space of dimension d under a locally adaptive normal distribution (LAND) metric """

    def __init__(self, data, sigma=1., rho=0.1): 
        super().__init__(data.shape[1])

        self.data = data # \in R^{n x d}
        self.sigma = sigma
        self.rho = rho

    def metric_tensor_diagonal(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        return 1 / (self.locally_adaptive_normal_distribution(x) + self.rho)
    
    def locally_adaptive_normal_distribution(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        return torch.sum(self.phi(x)[:,:,None] * self.w(x), 1)

    def phi(self, x):
        """
        :param x: N x d
        :return: N x n
        """
        return torch.exp(- torch.norm(x[:,None] - self.data[None],2,-1) ** 2 / (2 * self.sigma)) 
    
    def w(self, x):
        """
        :param x: N x d
        :return: N x n x d
        """
        return (x[:,None] - self.data[None]) ** 2