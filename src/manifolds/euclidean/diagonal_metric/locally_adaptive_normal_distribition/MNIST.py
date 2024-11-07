from src.manifolds.euclidean.diagonal_metric.locally_adaptive_normal_distribition import LocallyAdaptiveNormalDistributionEuclidean

class MnistLandRbfEuclidean(LocallyAdaptiveNormalDistributionEuclidean):
    def __init__(self, digits, sigma=0.1, rho=1e-3): 
        # TODO load MNIST from digit list
        data = 5.
        super().__init__(data, sigma=sigma, rho=rho)
