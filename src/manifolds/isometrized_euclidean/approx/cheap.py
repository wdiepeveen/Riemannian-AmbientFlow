import torch

from src.curve_solvers.boundary_value.harmonic import HarmonicBoundaryValueCurveSolver
from src.curves.boundary_value.time_changed import TimeChangedBoundaryValueCurve
from src.manifolds.isometrized_euclidean.approx import l2IsometrizedApproxEuclidean
from src.time_flows.piecewise_linear import PiecewiseLinearTimeFlow

class Cheapl2IsometrizedApproxEuclidean(l2IsometrizedApproxEuclidean):
    def __init__(self, euclidean_manifold):
        super().__init__(euclidean_manifold)

    def loss_function(self, x, X):
        return self.euclidean.norm(x,X[:,None])

    def geodesic(self, x, y, t, num_intervals=10, num_sines=1):
        geodesic_solver = HarmonicBoundaryValueCurveSolver(x, y, num_sines, self.loss_function)
        geodesic_solver.solve()
        
        # remetrize geodesic
        z = geodesic_solver.curve(torch.linspace(0., 1., num_intervals+1))
        time_flow = PiecewiseLinearTimeFlow(z)
        iso_geodesic = TimeChangedBoundaryValueCurve(geodesic_solver.curve, time_flow)

        return iso_geodesic(t).detach()
    
    def log(self, x, y, num_intervals=10, num_sines=1):
        N, _ = y.shape 
        logs = torch.zeros_like(y)
        for i in range(N):
            geodesic_solver = HarmonicBoundaryValueCurveSolver(x, y[i], num_sines, self.loss_function)
            geodesic_solver.solve()
            
            # remetrize geodesic
            z = geodesic_solver.curve(torch.linspace(0., 1., num_intervals+1))
            time_flow = PiecewiseLinearTimeFlow(z)
            iso_geodesic = TimeChangedBoundaryValueCurve(geodesic_solver.curve, time_flow)
            logs[i] = iso_geodesic.differential_forward(torch.zeros(1))
        return logs.detach()