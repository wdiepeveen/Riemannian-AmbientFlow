from src.manifolds import Manifold

class Euclidean(Manifold):
    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :param x: N x [Epoint]
        :return: N x [Evector] x [Evector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        :param x: N x [Epoint]
        :return: N x [Evector] x [Evector]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def gradient_metric_tensor(self, x):
        """
        g_ab;c
        :param x: N x [Epoint]
        :return: N x [(3,0)-Etensor]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def christoffel_symbols(self,x):
        """
        G^c_ab
        :param x: N x [Epoint]
        :return: N x [(2,1)-Etensor] 
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def curvature_tensor(self,x): 
        """
        R^d_cab
        :param x: N x [Epoint]
        :return: N x [(3,1)-Etensor]
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def ricci_tensor(self,x):
        """
        R_ab
        :param x: N x [Epoint]
        :return: N x [(2,0)-Etensor] 
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def ricci_scalar(self,x):
        """
        R
        :param x: N x [Epoint]
        :return: N 
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )