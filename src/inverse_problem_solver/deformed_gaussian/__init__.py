from src.inverse_problem_solver import InverseProblemSolver
from src.strongly_convex.learnable_diagonal_quadratic import LearnableDiagonalQuadratic

class DeformedGaussianSolver(InverseProblemSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd):
        super().__init__(diffeomorphism.d, forward_operator)
        self.phi = diffeomorphism
        self.psi = LearnableDiagonalQuadratic(self.d, use_softplus=True)
        self.lambd = lambd

    def reconstruct(self, y):
        return self.decode_to_data(self.encode(y))

    def decode_to_corrupted_data(self, z):
        return self.A.forward(self.phi.inverse(z))
    
    def decode_to_data(self, z):
        return self.phi.inverse(z)
    
    def encode(self, y): 
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def reconstruction_loss(self,y):
        z = self.encode(y)
        return 0.5 * ((self.decode_to_corrupted_data(z) - y)**2).mean(0).sum() + self.lambd * (self.psi.forward(z).mean() + 0.5 * self.psi.diagonal.log().sum())