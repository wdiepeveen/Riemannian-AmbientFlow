import torch

from src.manifolds.euclidean.pullback.deformed_gaussian import DeformedGaussianPullbackEuclidean
from src.unimodal.deformed_gaussian.quadratic_unbend import QuadraticUnbend

class QuadraticUnbendPullbackEuclidean(DeformedGaussianPullbackEuclidean):

    def __init__(self, delta=1., a1=1/4, a2=4):
        super().__init__(QuadraticUnbend(delta, torch.tensor([a1, a2])))