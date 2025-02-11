from src.inverse_problem_solver.encoder_decoder.deformed_anisotropic_gaussian import DeformedGaussianSolver

class UnrolledDeformedGaussianSolver(DeformedGaussianSolver):
    def __init__(self, forward_operator, diffeomorphism, sparsity_level, lambd, n_steps):
        super().__init__(forward_operator, diffeomorphism, sparsity_level, lambd)
        self.D = n_steps