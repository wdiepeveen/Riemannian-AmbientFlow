from src.riemannian_autoencoder.image import ImageRiemannianAutoencoder

class PullbackImageRiemannianAutoencoder(ImageRiemannianAutoencoder):
    def __init__(self, pullback_image_euclidean):
        super().__init__(pullback_image_euclidean)

