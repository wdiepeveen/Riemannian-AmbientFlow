from src.riemannian_autoencoder import RiemannianAutoencoder

class VectorRiemannianAutoencoder(RiemannianAutoencoder):
    def __init__(self, vector_euclidean):
        super().__init__(vector_euclidean)