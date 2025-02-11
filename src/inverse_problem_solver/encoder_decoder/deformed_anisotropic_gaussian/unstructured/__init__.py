from src.inverse_problem_solver.encoder_decoder.deformed_anisotropic_gaussian import DeformedGaussianSolver

class UnstructuredDeformedGaussianSolver(DeformedGaussianSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd, encoder, sparity_level):
        super().__init__(forward_operator, diffeomorphism, sparity_level, lambd)
        self.encoder = encoder
        self.k = sparity_level

    def encode(self, y): 
        return self.encoder.forward(y)