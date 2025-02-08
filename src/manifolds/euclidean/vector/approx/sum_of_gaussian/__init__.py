import torch

from manifolds.euclidean.vector.approx import ApproxVectorEuclidean

class SumOfGaussianApproxVectorEuclidean(ApproxVectorEuclidean): 
    """ Base class describing Euclidean space of dimension d under a sum of gaussian metric """

    def __init__(self, sum_of_gaussian):
        super().__init__(sum_of_gaussian.d)
        
        self.mumo = sum_of_gaussian
        self.psi = self.mumo.psi
        self.weights = self.mumo.weights
        self.m = self.mumo.m

        self.inverse_matrices = torch.cat([self.psi[i].inverse_matrix[None] for i in range(self.m)])
        self.multiplied_inverse_matrices = torch.einsum("mab,nbc->mnac", self.inverse_matrices, self.inverse_matrices)

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        N, _ = x.shape
        psi_x = torch.zeros(N,self.m)
        for i in range(self.m):
            psi_x[:,i] = self.psi[i].forward(x)
        softmax_psi_x = (- psi_x + torch.log(self.weights[None] + 1e-8)).softmax(1)
        return torch.sum( softmax_psi_x[:,:,None,None,None] * softmax_psi_x[:,None,:,None,None] * self.multiplied_inverse_matrices[None], [1,2])
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x d x d
        """
        return torch.inverse(self.metric_tensor(x))
