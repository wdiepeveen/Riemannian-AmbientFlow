import torch

from src.manifolds.euclidean import Euclidean

class VectorEuclidean(Euclidean):
    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inverse_metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def gradient_metric_tensor(self, x):
        """
        g_ab;c
        :param x: N x d
        :return: N x d x d x d
        """
        def sum_metric_tensor(y):
            return torch.sum(self.metric_tensor(y),0)
        
        return torch.autograd.functional.jacobian(sum_metric_tensor, x).permute((2,0,1,3))
    
    def christoffel_symbols(self,x):
        """
        G^c_ab
        :param x: N x d
        :return: N x d x d x d 
        """
        inverse_metric_tensor_x = self.inverse_metric_tensor(x)
        gradient_metric_tensor_x = self.gradient_metric_tensor(x)

        term_1 = torch.einsum("Ncd,Ndab->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        term_2 = torch.einsum("Ncd,Ndba->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        term_3 = torch.einsum("Ncd,Nabd->Nabc",inverse_metric_tensor_x, gradient_metric_tensor_x)
        return 1/2 * (term_1 + term_2 - term_3)
    
    def curvature_tensor(self,x): 
        """
        R^d_cab
        :param x: N x d
        :return: N x d x d x d x d
        """
        def sum_christoffel_symbol(y):
            return torch.sum(self.christoffel_symbols(y),0)
        
        christoffel_symbol_gradient_x = torch.autograd.functional.jacobian(sum_christoffel_symbol, x).permute((3,4,0,1,2))
        christoffel_symbols_x = self.christoffel_symbols(x)
        christoffel_symbol_product_x = torch.einsum("Naed,Nbce->Nabcd", christoffel_symbols_x, christoffel_symbols_x)

        term_1 = christoffel_symbol_gradient_x
        term_2 = christoffel_symbol_gradient_x.permute(0,2,1,3,4)
        term_3 = christoffel_symbol_product_x
        term_4 = christoffel_symbol_product_x.permute(0,2,1,3,4)

        return term_1 - term_2 + term_3 - term_4
    
    def ricci_tensor(self,x):
        """
        R_ab
        :param x: N x d
        :return: N x d x d 
        """
        curvature_tensor_x = self.curvature_tensor(x)
        return torch.einsum("Ncabc->Nab", curvature_tensor_x)
    
    def ricci_scalar(self,x):
        """
        R
        :param x: N x d
        :return: N 
        """
        ricci_tensor_x = self.ricci_tensor(x)
        return torch.einsum("Naa->N", ricci_tensor_x)