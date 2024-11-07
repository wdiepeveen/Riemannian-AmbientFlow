import torch

from src.curves.boundary_value.discrete import DiscreteBoundaryValueCurve
from src.curves.boundary_value.time_changed import TimeChangedBoundaryValueCurve
from src.manifolds.isometrized_euclidean import l2IsometrizedEuclidean
from src.time_flows.piecewise_linear import PiecewiseLinearTimeFlow

class l2IsometrizedExactEuclidean(l2IsometrizedEuclidean):
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

    def barycentre(self, x, tol=1e-2, max_iter=100, step_size=1/4): 
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
    
    def geodesic(self, x, y, t, num_intervals=100):
        """
        Discrete geodesic approximation
        :param x: d
        :param y: d
        :param t: N
        :return: N x d
        """
        geodesic = DiscreteBoundaryValueCurve(x, y, num_intervals)
        geodesic.coefficients = torch.nn.Parameter(self.euclidean.geodesic(x,y,torch.linspace(0.,1.,num_intervals+1))[1:-1])
        
        # remetrize geodesic
        z = torch.cat([x[None], geodesic.coefficients, y[None]], dim=0)
        time_flow = PiecewiseLinearTimeFlow(z)
        iso_geodesic = TimeChangedBoundaryValueCurve(geodesic, time_flow)

        return iso_geodesic(t).detach()

    def log(self, x, y, num_intervals=100):
        """
        Discrete geodesic approximation
        :param x: d
        :param y: N x d
        :return: N x d
        """
        N, _ = y.shape 
        logs = torch.zeros_like(y)
        for i in range(N):
            geodesic = DiscreteBoundaryValueCurve(x, y[i], num_intervals)
            geodesic.coefficients = torch.nn.Parameter(self.euclidean.geodesic(x,y[i],torch.linspace(0.,1.,num_intervals+1))[1:-1])
        
            # remetrize geodesic
            z = torch.cat([x[None], geodesic.coefficients, y[i][None]], dim=0)
            time_flow = PiecewiseLinearTimeFlow(z)
            iso_geodesic = TimeChangedBoundaryValueCurve(geodesic, time_flow)
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
                y_k = self.euclidean.geodesic(z[-2], z[-1], torch.tensor([2.]))[0]
                z.append(y_k)

                # compute c_k
                path_length += torch.linalg.norm(z[-1] - z[-2],2)
            
            exps[i] = z[-1]
        return exps.detach()
    
    def distance(self, x, y, num_intervals=100):
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
                    geodesic = DiscreteBoundaryValueCurve(x[i,j], y[i,k], num_intervals)
                    geodesic.coefficients = torch.nn.Parameter(self.euclidean.geodesic(x[i,j],y[i,k],torch.linspace(0.,1.,num_intervals+1))[1:-1])
        
                    geodesic_t = geodesic(torch.linspace(0.,1.,200))
                    distances[i,j,k] = torch.sum(self.norm(geodesic_t[0:-1], (geodesic_t[1:] - geodesic_t[0:-1])[:,None]))
        return distances.detach()

    def parallel_transport(self, x, X, y, num_bv_intervals=100, num_iv_intervals=200): 
        """
        Pole ladder approximation
        :param x: d
        :param X: N x d
        :param y: d
        :return: N x d
        """
        m = self.geodesic(x, y, 0.5 * torch.ones(1), num_intervals=num_bv_intervals)[0]
        p = self.exp(x, X, num_intervals=num_iv_intervals)
        D = - self.log(m, p, num_intervals=num_bv_intervals)
        q = self.exp(m, D, num_intervals=num_iv_intervals)
        Y = - self.log(y, q, num_intervals=num_bv_intervals)
        return Y
