from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.vector import VectorDiffeomorphism
        
class NFlowVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, d, nflow):
        super().__init__(d)

        self.nflow = nflow

    def forward(self, x, context=None):
        """
        Forward pass through the diffeomorphism.
        :param x: N x d
        :return: N x d
        """
        out, log_abs_det = self.nflow._transform(x, context=context)
        return out

    def inverse(self, y, context=None):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x d
        :return: N x d
        """
        out, log_abs_det = self.nflow._transform.inverse(y, context=context)
        return out

    def differential_forward(self, x, X, context=None):
        """
        Compute the differential map of phi at x for a vector X.
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        _, jvp_result = jvp(lambda x: self.nflow._transform(x, context=context)[0], (x,), (X,))
        return jvp_result

    def differential_inverse(self, y, Y, context=None):
        """
        Compute the differential map of the inverse of phi at y for a vector Y.
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        _, jvp_result = jvp(lambda y: self.nflow._transform.inverse(y, context=context)[0], (y,), (Y,))
        return jvp_result
    
    def adjoint_differential_forward(self, x, X, context=None):
        """
        Compute the adjoint differential map of phi at x for a vector X.
        
        :param x: N x d
        :param X: N x d
        :return: N x d
        """
        _, vjp_result = vjp(lambda x: self.nflow._transform(x, context=context)[0], x, X)
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y, context=None):
        """
        Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
        :param y: N x d
        :param Y: N x d
        :return: N x d
        """
        _, vjp_result = vjp(lambda y: self.nflow._transform.inverse(y, context=context)[0], (y,), (Y,))
        return vjp_result[0]
    
