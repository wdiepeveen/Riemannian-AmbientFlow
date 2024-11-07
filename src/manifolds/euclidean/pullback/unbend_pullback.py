import torch

from src.diffeomorphisms.unbend import UnbendDiffeomorphism
from src.manifolds.euclidean.pullback import PullbackEuclidean

class UnbendPullbackEuclidean(PullbackEuclidean):

    def __init__(self, angle=torch.pi/4, delta=1.):
        super().__init__(UnbendDiffeomorphism(angle, delta))