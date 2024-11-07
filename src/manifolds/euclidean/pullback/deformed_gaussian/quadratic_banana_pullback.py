import torch

from src.manifolds.euclidean.pullback.deformed_gaussian import DeformedGaussianPullbackEuclidean
from src.unimodal.deformed_gaussian.quadratic_banana import QuadraticBanana

class QuadraticBananaPullbackEuclidean(DeformedGaussianPullbackEuclidean):

    def __init__(self, shear=1/9, offset=0., a1=1/4, a2=4):
        super().__init__(QuadraticBanana(shear, offset, torch.tensor([a1, a2])))