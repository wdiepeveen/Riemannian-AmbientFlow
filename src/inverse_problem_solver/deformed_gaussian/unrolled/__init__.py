from src.inverse_problem_solver.deformed_gaussian import DeformedGaussianSolver

class UnrolledDeformedGaussianSolver(DeformedGaussianSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd, n_steps):
        super().__init__(forward_operator, diffeomorphism, lambd)
        self.D = n_steps