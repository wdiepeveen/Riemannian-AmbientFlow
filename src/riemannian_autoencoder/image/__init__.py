from src.riemannian_autoencoder import RiemannianAutoencoder

class ImageRiemannianAutoencoder(RiemannianAutoencoder):
    def __init__(self, image_euclidean, device):
        super().__init__(image_euclidean, device)
        self.C = image_euclidean.C
        self.H = image_euclidean.H
        self.W = image_euclidean.W