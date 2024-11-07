import torch

from src.manifolds import Manifold


class Euclidean(Manifold):
    def __init__(self, d):
        super().__init__(d)

    def metric_tensor(self, x):
        """
        :return: N x d x d
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def inner(self, x, X, Y):
        """

        :param x: N x Mpoint
        :param X: N x M x Mvector
        :param Y: N x L x Mvector
        :return: N x M x L
        """
        metric_tensor_x = self.metric_tensor(x)
        return torch.einsum("Nab,NMa,NLb->NML", metric_tensor_x, X, Y)
    
    # def barycentre(self, x, tol=1e-2, max_iter=50, step_size=1/4): # TODO allow to recycle discrete geodesics + move to Euclidean __init__
    #     """

    #     :param x: N x d
    #     :return: d
    #     """
    #     k = 0
    #     y = torch.mean(x,0)
        
    #     gradient_0 = torch.mean(self.log(y, x),0)
    #     error = self.norm(y[None], gradient_0[None,None]) + 1e-2
    #     rel_error = 1.
    #     while k <= max_iter and rel_error >= tol:
    #         gradient = torch.mean(self.log(y, x),0)
    #         y = y + step_size * gradient
    #         k+=1
    #         rel_error = self.norm(y[None], gradient[None,None]) / error
    #         print(f"iteration {k} | rel_error = {rel_error.item()}")

    #     print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k} iterations")

    #     return y

    
    # def geodesic(self, x, y, t, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param y: d
    #     :param t: N
    #     :return: N x d
    #     """
    #     geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x, y, self.norm, 
    #                                                           num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr,
    #                                                           initialize=initialize, num_sines=num_sines)
    #     geodesic_solver.solve()
    #     return geodesic_solver.geodesic(t).detach()

    # def log(self, x, y, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param y: N x d
    #     :return: N x d
    #     """
    #     N, _ = y.shape 
    #     logs = torch.zeros_like(y)
    #     for i in range(N):
    #         geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x, y[i], self.norm, 
    #                                                               num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, 
    #                                                               initialize=initialize, num_sines=num_sines)
    #         geodesic_solver.solve()
    #         logs[i] = geodesic_solver.geodesic.differential_forward(torch.zeros(1))
    #     return logs.detach()

    # def exp(self, x, X, num_intervals=200):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param X: N x d
    #     :return: N x d
    #     """
    #     N, _ = X.shape 
    #     exps = torch.zeros_like(X)
    #     for i in range(N):
    #         geodesic_solver = DiscreteInitialValueGeodesicSolver(x, X[i], self.norm, self.metric_tensor, self.gradient_metric_tensor, num_intervals=num_intervals)
    #         geodesic_solver.solve()
    #         exps[i] = geodesic_solver.geodesic(torch.ones(1))
    #     return exps.detach()
    
    # def distance(self, x, y, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
    #     """
    #     Summed segment length of discrete geodesic approximation
    #     :param x: N x M x d
    #     :param y: N x L x d
    #     :return: N x M x L
    #     """
    #     N, M, _ = x.shape 
    #     N, L, _ = y.shape 
    #     distances = torch.zeros(N,M,L)
    #     for i in range(N):
    #         for j in range(M):
    #             for k in range(L):
    #                 geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x[i,j], y[i,k], self.norm, 
    #                                                                       num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, 
    #                                                                       initialize=initialize, num_sines=num_sines)
    #                 geodesic_solver.solve()
    #                 geodesic_t = geodesic_solver.geodesic(torch.linspace(0.,1.,200))
    #                 distances[i,j,k] = torch.sum(self.norm(geodesic_t[0:-1], (geodesic_t[1:] - geodesic_t[0:-1])[:,None]))
    #     return distances.detach()

    # def parallel_transport(self, x, X, y, num_bv_intervals=10, num_bv_time_points=200, num_bv_epochs=1000, bv_lr=1e-4, num_iv_intervals=200, initialize=True, num_sines=1): # TODO add iv parameters
    #     """
    #     Pole ladder approximation
    #     :param x: d
    #     :param X: N x d
    #     :param y: d
    #     :return: N x d
    #     """
    #     m = self.geodesic(x, y, 0.5 * torch.ones(1), 
    #                       num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
    #                       initialize=initialize, num_sines=num_sines)[0]
    #     p = self.exp(x, X, num_intervals=num_iv_intervals)
    #     D = - self.log(m, p, 
    #                    num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
    #                    initialize=initialize, num_sines=num_sines)
    #     q = self.exp(m, D, num_intervals=num_iv_intervals)
    #     Y = - self.log(y, q, 
    #                    num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
    #                    initialize=initialize, num_sines=num_sines)
    #     return Y
    
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