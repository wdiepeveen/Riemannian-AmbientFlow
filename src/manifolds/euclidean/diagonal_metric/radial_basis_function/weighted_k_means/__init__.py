import torch

from src.manifolds.euclidean.diagonal_metric.radial_basis_function import RadialBasisFunctionEuclidean
from src.radial_basis_functions.weighted_k_means import WeightedKMeansRbf

class WeightedKMeansRBFEuclidean(RadialBasisFunctionEuclidean):
    """ Base class describing Euclidean space of dimension d under a weighted K-means metric """

    def __init__(self, data, K_means=10, kappa=1., lambd=1e-2, rho=1e-3): 
        super().__init__(data.shape[1], rho)

        self.data = data # \in R^{n x d}

        self.rbf = WeightedKMeansRbf(self.d, K_means=K_means, kappa=kappa, lambd=lambd)
        print("Fitting data")
        self.rbf.fit(data) 
    
    def radial_basis_function(self, x):
        return self.rbf(x)