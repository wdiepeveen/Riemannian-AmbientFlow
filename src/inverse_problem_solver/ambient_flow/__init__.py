from src.inverse_problem_solver import InverseProblemSolver

class AmbientFlowSolver(InverseProblemSolver):
    def __init__(self, d, forward_operator, fixed_origin_diffeomorphism, conditional_diffeomorphism, sparsity_level):
        super().__init__(d, forward_operator)
        self.phi = fixed_origin_diffeomorphism
        self.phi_cond = conditional_diffeomorphism
        self.k = sparsity_level