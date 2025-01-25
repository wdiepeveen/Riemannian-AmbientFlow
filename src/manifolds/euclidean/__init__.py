import torch
from src.manifolds import Manifold


class Euclidean(Manifold):
    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    