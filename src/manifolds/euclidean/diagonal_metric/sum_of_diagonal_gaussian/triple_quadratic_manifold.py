import torch

from src.manifolds.euclidean.diagonal_metric.sum_of_diagonal_gaussian import SumOfDiagonalEuclidean
from src.multimodal.sum_of_diagonal_gaussian import SumOfDiagonalGaussian

class TripleDiagonalGaussianEuclidean(SumOfDiagonalEuclidean):

    def __init__(self, a1=1/4, a2=4, offset=5.):
        super().__init__(SumOfDiagonalGaussian(torch.tensor([[a1, a2], [a2, a1], [a1, a2]]), 
                                               torch.tensor([[-offset, -offset], [0.,0.], [offset, offset]]), 
                                               torch.tensor([1., 1., 1.])))