import torch

from src.curves.boundary_value.time_changed import TimeChangedBoundaryValueCurve
from src.geodesic_solvers.boundary_value.discrete import DiscreteBoundaryValueGeodesicSolver
from src.geodesic_solvers.initial_value.discrete import DiscreteInitialValueGeodesicSolver
from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean
from src.time_flows.piecewise_linear import PiecewiseLinearTimeFlow

class l2IsometrizedApproxEuclidean(l2IsometrizedEuclidean):
    def __init__(self, euclidean_manifold):
        super().__init__(euclidean_manifold)

        self.euclidean = euclidean_manifold

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)

    def barycentre(self, x, tol=1e-2, max_iter=50, step_size=1/4): 
        """

        :param x: N x d
        :return: d
        """
        k = 0
        y = torch.mean(x,0)
        
        gradient_0 = torch.mean(self.log(y, x),0)
        error = self.norm(y[None], gradient_0[None,None]) + 1e-2
        rel_error = 1.
        while k <= max_iter and rel_error >= tol:
            gradient = torch.mean(self.log(y, x),0)
            y = y + step_size * gradient
            k+=1
            rel_error = self.norm(y[None], gradient[None,None]) / error
            print(f"iteration {k} | rel_error = {rel_error.item()}")

        print(f"gradient descent was terminated after reaching a relative error {rel_error.item()} in {k} iterations")

        return y
    
    def geodesic(self, x, y, t, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
        """
        Discrete geodesic approximation
        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x, y, self.euclidean.norm, 
                                                              num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr,
                                                              initialize=initialize, num_sines=num_sines)
        geodesic_solver.solve()
        
        # remetrize geodesic
        z = torch.cat([x[None], geodesic_solver.geodesic.coefficients, y[None]], dim=0)
        time_flow = PiecewiseLinearTimeFlow(z)
        iso_geodesic = TimeChangedBoundaryValueCurve(geodesic_solver.geodesic, time_flow)

        return iso_geodesic(t).detach()

    def log(self, x, y, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
        """
        Discrete geodesic approximation
        :param x: d
        :param y: N x d
        :return: N x d
        """
        N, _ = y.shape 
        logs = torch.zeros_like(y)
        for i in range(N):
            geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x, y[i], self.euclidean.norm, 
                                                                  num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, 
                                                                  initialize=initialize, num_sines=num_sines)
            geodesic_solver.solve()
            
            # remetrize geodesic
            z = torch.cat([x[None], geodesic_solver.geodesic.coefficients, y[i][None]], dim=0)
            time_flow = PiecewiseLinearTimeFlow(z)
            iso_geodesic = TimeChangedBoundaryValueCurve(geodesic_solver.geodesic, time_flow)
            logs[i] = iso_geodesic.differential_forward(torch.zeros(1))
        return logs.detach()

    def exp(self, x, X, num_intervals=200):
        """
        Discrete geodesic approximation
        :param x: d
        :param X: N x d
        :return: N x d
        """
        N, _ = X.shape 
        exps = torch.zeros_like(X)
        for i in range(N):
            z = [x, x + 1/num_intervals * X[i]]

            path_length = torch.linalg.norm(1/num_intervals * X[i],2)
            X_norm =  torch.linalg.norm(X[i],2)

            while path_length < X_norm:
                # compute y_k
                geodesic_solver = DiscreteInitialValueGeodesicSolver(z[-2], 2 * (z[-1] - z[-2]), self.euclidean.norm, self.euclidean.metric_tensor, self.euclidean.gradient_metric_tensor, num_intervals=2)
                geodesic_solver.solve()
                y_k = geodesic_solver.geodesic(torch.ones(1))[0]
                z.append(y_k)

                # compute c_k
                path_length += torch.linalg.norm(z[-1] - z[-2],2)
            
            exps[i] = z[-1]
        return exps.detach()
    
    def distance(self, x, y, num_intervals=10, num_time_points=200, num_epochs=1000, lr=1e-4, initialize=True, num_sines=1):
        """
        Summed segment length of discrete geodesic approximation
        :param x: N x M x d
        :param y: N x L x d
        :return: N x M x L
        """
        N, M, _ = x.shape 
        N, L, _ = y.shape 
        distances = torch.zeros(N,M,L)
        for i in range(N):
            for j in range(M):
                for k in range(L):
                    geodesic_solver = DiscreteBoundaryValueGeodesicSolver(x[i,j], y[i,k], self.euclidean.norm, 
                                                                          num_intervals=num_intervals, num_time_points=num_time_points, num_epochs=num_epochs, lr=lr, 
                                                                          initialize=initialize, num_sines=num_sines)
                    geodesic_solver.solve()
                    geodesic_t = geodesic_solver.geodesic(torch.linspace(0.,1.,200))
                    distances[i,j,k] = torch.sum(self.norm(geodesic_t[0:-1], (geodesic_t[1:] - geodesic_t[0:-1])[:,None]))
        return distances.detach()

    def parallel_transport(self, x, X, y, num_bv_intervals=10, num_bv_time_points=200, num_bv_epochs=1000, bv_lr=1e-4, num_iv_intervals=200, initialize=True, num_sines=1): 
        """
        Pole ladder approximation
        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        m = self.geodesic(x, y, 0.5 * torch.ones(1), 
                          num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
                          initialize=initialize, num_sines=num_sines)[0]
        p = self.exp(x, X, num_intervals=num_iv_intervals)
        D = - self.log(m, p, 
                       num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
                       initialize=initialize, num_sines=num_sines)
        q = self.exp(m, D, num_intervals=num_iv_intervals)
        Y = - self.log(y, q, 
                       num_intervals=num_bv_intervals, num_time_points=num_bv_time_points, num_epochs=num_bv_epochs, lr=bv_lr,
                       initialize=initialize, num_sines=num_sines)
        return Y
