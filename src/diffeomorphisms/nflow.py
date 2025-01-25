import torch

from src.diffeomorphisms import Diffeomorphism

from torch.autograd.functional import jvp, vjp


        
class NFlowDiffeomorphism(Diffeomorphism):
    def __init__(self, d, nflow):
        super().__init__(d)

        self.diffeo = nflow

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: Input tensor, shape (B, C, H, W) or (B, D)
        :return: Transformed tensor
        """
        out, logabsdetjacobian = self.diffeo._transform(x, context=None)
        return out

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: Input tensor, shape (B, C, H, W) or (B, D)
        :return: Inverse-transformed tensor
        """
        out, logabsdetjacobian = self.diffeo._transform.inverse(y, context=None)
        return out

    def differential_forward(self, x, X):
        """
        Compute the differential map of phi at x for a vector X.
        
        :param x: A batch of points, N x 2.
        :param X: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """
        _, jvp_result = jvp(lambda x: self.diffeo._transform(x, context=None)[0], (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        
        :param y: A batch of points, N x 2.
        :param Y: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """
        _, jvp_result = jvp(lambda y: self.diffeo._transform.inverse(y, context=None)[0], (y,), (Y,))
        return jvp_result
    
    def adjoint_differential_forward(self, x, X):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: A batch of points, N x 2.
        :param X: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """
        _, vjp_result = vjp(lambda x: self.diffeo._transform(x, context=None)[0], x, X)
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: A batch of points, N x 2.
        :param Y: A batch of tangent vectors, N x 2.
        :return: A batch of transformed tangent vectors, N x 2.
        """
        _, vjp_result = vjp(lambda y: self.diffeo._transform.inverse(y, context=None)[0], (y,), (Y,))
        return vjp_result[0]
    
