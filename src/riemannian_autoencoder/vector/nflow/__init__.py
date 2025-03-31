from src.riemannian_autoencoder.vector import VectorRiemannianAutoencoder

class NFlowVectorRiemannianAutoencoder(VectorRiemannianAutoencoder):
    def __init__(self, pullback_vector_euclidean):
        super().__init__(pullback_vector_euclidean)

