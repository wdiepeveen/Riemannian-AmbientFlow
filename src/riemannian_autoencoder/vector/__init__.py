from src.riemannian_autoencoder import RiemannianAutoencoder

class VectorRiemannianAutoencoder(RiemannianAutoencoder):
    def __init__(self, vector_euclidean, device):
        super().__init__(vector_euclidean, device)