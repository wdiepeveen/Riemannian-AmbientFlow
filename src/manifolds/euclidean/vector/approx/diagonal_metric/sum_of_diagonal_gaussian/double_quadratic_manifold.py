import torch

from src.manifolds.euclidean.vector.approx.diagonal_metric.sum_of_diagonal_gaussian import SumOfDiagonalGaussianApproxVectorEuclidean
from src.multimodal.sum_of_diagonal_gaussian import SumOfDiagonalGaussian

class DoubleDiagonalGaussianApproxVectorEuclidean(SumOfDiagonalGaussianApproxVectorEuclidean):

    def __init__(self, a1=1/4, a2=4, a3=4, a4=1/4, offset=0, w1=1, w2=1):
        super().__init__(SumOfDiagonalGaussian(torch.tensor([[a1, a2], [a3, a4]]), 
                                               torch.tensor([[-offset, 0.], [0., offset]]), 
                                               torch.tensor([w1, w2])))