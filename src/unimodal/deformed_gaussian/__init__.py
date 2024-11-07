from src.unimodal import Unimodal
from src.strongly_convex.diagonal_quadratic import DiagonalQuadratic

class DeformedGaussian(Unimodal):
    def __init__(self, diffeomorphism, diagonal) -> None:
        super().__init__(diffeomorphism, DiagonalQuadratic(diagonal)) 

