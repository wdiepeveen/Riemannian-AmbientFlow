import torch
import torch.nn as nn

from src.inverse_problem_solver.encoder_decoder.deformed_anisotropic_gaussian.unrolled import UnrolledDeformedGaussianSolver

class GradientDescentUnrolledSolver(UnrolledDeformedGaussianSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd, n_steps, feed_forward_net):
        super().__init__(forward_operator, diffeomorphism, lambd, n_steps)
        self.F = feed_forward_net

    def encode(self, y):
        N, _ = y.shape
        z = torch.zeros(N, self.d, device=y.device)
        for _ in range(self.D):
            data_term = self.phi.adjoint_differential_inverse(
                z, 
                self.A.adjoint_forward(self.A.forward(self.phi.inverse(z)) - y)
            )
            reg_term = self.lambd * self.psi.grad_forward(z)
            input_tensor = torch.cat([z, data_term + reg_term], dim=1)
            z = z - self.F.forward(input_tensor)
        return z