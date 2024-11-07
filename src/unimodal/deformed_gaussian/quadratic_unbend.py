from src.unimodal.deformed_gaussian import DeformedGaussian
from src.diffeomorphisms.unbend import UnbendDiffeomorphism

class QuadraticUnbend(DeformedGaussian):
    def __init__(self, delta, diagonal) -> None:
        super().__init__(UnbendDiffeomorphism(0., delta), diagonal)

