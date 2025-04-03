from src.dimension_reduction.principal_geodesic_analysis.vector.l2_tangent_space_pca import l2TangentSpacePCAVectorSolver
from src.riemannian_autoencoder.vector.pullback import PullbackVectorRiemannianAutoencoder

class l2TangentSpacePCAPullbackVectorRiemannianAutoencoder(PullbackVectorRiemannianAutoencoder):
    def __init__(self, pullback_vector_euclidean, data, latent_dim):
        super().__init__(pullback_vector_euclidean)
        self.data = data
        self.base_point = self.manifold.barycentre()
        self.m = latent_dim

        self.dim_red = l2TangentSpacePCAVectorSolver(self.data, self.manifold, self.base_point)

        
