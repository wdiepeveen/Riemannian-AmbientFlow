import torch
from src.manifolds.euclidean.pullback.deformed_gaussian.quadratic_banana_pullback import QuadraticBananaPullbackEuclidean
from src.unimodal import Unimodal
from src.unimodal.deformed_gaussian.quadratic_river import QuadraticRiver
from src.manifolds.euclidean.pullback.deformed_gaussian import DeformedGaussianPullbackEuclidean

def get_ground_truth_pullback_manifold(config):
    if config.dataset == 'single_banana':
        pullback_manifold = QuadraticBananaPullbackEuclidean()
    elif config.dataset == 'river':
        shear, offset, a1, a2 = 2, 0, 1/25, 3
        river_unimodal_distribution = QuadraticRiver(shear, offset, torch.tensor([a1, a2]))
        pullback_manifold = DeformedGaussianPullbackEuclidean(river_unimodal_distribution)

    return pullback_manifold

def get_learned_pullback_manifold(phi, psi):
    distribution = Unimodal(diffeomorphism=phi, strongly_convex=psi)
    manifold = DeformedGaussianPullbackEuclidean(distribution)
    return manifold
