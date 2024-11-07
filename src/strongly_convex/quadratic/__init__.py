import torch

from src.strongly_convex import StronglyConvex

class Quadratic(StronglyConvex): 
    """ Class that implements the strongly convex function x \mapsto 1/2 x^\top A^{-1} x, where A is SPD """
    def __init__(self, matrix, offset=None, inverse=False) -> None:
        super().__init__(len(matrix))

        if inverse == False:
            self.inverse_matrix = torch.inverse(matrix)
        else:
            self.inverse_matrix = matrix
        if offset is None:
            self.offset = torch.zeros(self.d)
        else:
            self.offset = offset

    def forward(self, x):
        """
        :param x: N x d
        :return: N
        """
        return 0.5 * torch.einsum("ab,Na,Nb->N", self.inverse_matrix, x - self.offset[None], x - self.offset[None])
    
    def grad_forward(self, x):
        """
        :param x: N x d
        :return: N x d
        """
        return torch.einsum("ab,Nb->Na", self.inverse_matrix, x - self.offset[None])
    
    def differential_grad_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        return torch.einsum("ab,Nb->Na", self.inverse_matrix, X)