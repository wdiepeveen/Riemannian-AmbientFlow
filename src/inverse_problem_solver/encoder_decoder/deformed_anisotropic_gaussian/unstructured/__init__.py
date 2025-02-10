from src.inverse_problem_solver.encoder_decoder.deformed_anisotropic_gaussian import DeformedGaussianSolver

class UnstructuredDeformedGaussianSolver(DeformedGaussianSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd, encoder):
        super().__init__(forward_operator, diffeomorphism, lambd)
        self.encoder = encoder

    def encode(self, y): 
        return self.encoder.forward(y)