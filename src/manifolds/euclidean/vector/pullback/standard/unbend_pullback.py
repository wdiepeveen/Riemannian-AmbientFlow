import torch

from src.diffeomorphisms.vector.unbend import UnbendVectorDiffeomorphism
from src.manifolds.euclidean.vector.pullback.standard import StandardPullbackVectorEuclidean

class UnbendStandardPullbackVectorEuclidean(StandardPullbackVectorEuclidean):

    def __init__(self, angle=torch.pi/4, delta=1.):
        super().__init__(UnbendVectorDiffeomorphism(angle, delta))