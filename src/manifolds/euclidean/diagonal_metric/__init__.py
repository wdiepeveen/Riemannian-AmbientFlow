import torch

from src.manifolds.euclidean import Euclidean

class DiagonalMetricEuclidean(Euclidean):
    """ Base class describing Euclidean space of dimension d under a diagonal metric """

    def __init__(self, d): 
        super().__init__(d)
    
    def metric_tensor(self, x): 
        """
        :param x: N x d
        :return: N x d x d 
        """
        return torch.diag_embed(self.metric_tensor_diagonal(x))
        
    def inverse_metric_tensor(self, x):
        """
        :param x: N x d
        :return: N x d x d 
        """
        return torch.diag_embed(1 / self.metric_tensor_diagonal(x))
    
    def metric_tensor_diagonal(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    