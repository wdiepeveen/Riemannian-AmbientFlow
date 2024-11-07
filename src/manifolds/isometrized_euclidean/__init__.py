import torch

from src.curves.boundary_value.discrete import DiscreteBoundaryValueCurve
from src.curves.boundary_value.time_changed import TimeChangedBoundaryValueCurve
from src.geodesic_solvers.boundary_value.discrete import DiscreteBoundaryValueGeodesicSolver
from src.geodesic_solvers.initial_value.discrete import DiscreteInitialValueGeodesicSolver
from src.manifolds import Manifold
from src.time_flows.piecewise_linear import PiecewiseLinearTimeFlow

class l2IsometrizedEuclidean(Manifold):
    def __init__(self, euclidean_manifold):
        super().__init__(euclidean_manifold.d)

        self.euclidean = euclidean_manifold

    def inner(self, x, X, Y):
        """

        :param x: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        return torch.einsum("NMi,NLi->NML", X, Y)

    # def barycentre(self, x): 
    #     """

    #     :param x: N x d
    #     :return: d
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )
    
    # def geodesic(self, x, y, t):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param y: d
    #     :param t: N
    #     :return: N x d
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )

    # def log(self, x, y):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param y: N x d
    #     :return: N x d
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )

    # def exp(self, x, X):
    #     """
    #     Discrete geodesic approximation
    #     :param x: d
    #     :param X: N x d
    #     :return: N x d
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )
    
    # def distance(self, x, y):
    #     """
    #     Summed segment length of discrete geodesic approximation
    #     :param x: N x M x d
    #     :param y: N x L x d
    #     :return: N x M x L
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )

    # def parallel_transport(self, x, X, y): 
    #     """
    #     Pole ladder approximation
    #     :param x: d
    #     :param X: N x d
    #     :param y: d
    #     :return: N x d
    #     """
    #     raise NotImplementedError(
    #         "Subclasses should implement this"
    #     )
