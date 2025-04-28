from src.riemannian_autoencoder.vector import VectorRiemannianAutoencoder

class PullbackVectorRiemannianAutoencoder(VectorRiemannianAutoencoder):
    def __init__(self, pullback_vector_euclidean, device):
        super().__init__(pullback_vector_euclidean, device)

