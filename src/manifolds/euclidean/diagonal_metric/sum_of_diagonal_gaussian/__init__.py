import torch

from src.manifolds.euclidean.diagonal_metric import DiagonalMetricEuclidean

class SumOfDiagonalEuclidean(DiagonalMetricEuclidean):
    """ Base class describing Euclidean space of dimension d under a sum of diagonal gaussian metric """

    def __init__(self, sum_of_gaussian): 
        super().__init__(sum_of_gaussian.d)

        self.mumo = sum_of_gaussian
        self.psi = self.mumo.psi
        self.weights = self.mumo.weights
        self.m = self.mumo.m

        self.inverse_diagonals = torch.cat([self.psi[i].inverse_diagonal[None] for i in range(self.m)])
    
    def metric_tensor_diagonal(self, x):
        """
        :param x: N x d
        :return: N x d 
        """
        N, _ = x.shape
        psi_x = torch.zeros(N,self.m)
        for i in range(self.m):
            psi_x[:,i] = self.psi[i].forward(x)
        softmax_psi_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        return torch.sum((softmax_psi_x[:,:,None] * self.inverse_diagonals[None])[:,:,None,:] * (softmax_psi_x[:,:,None] * self.inverse_diagonals[None])[:,None,:,:],[1,2])