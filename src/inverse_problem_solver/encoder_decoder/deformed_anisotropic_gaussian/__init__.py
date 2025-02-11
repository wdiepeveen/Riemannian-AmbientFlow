from src.inverse_problem_solver.encoder_decoder import EncoderDecoderSolver
from src.strongly_convex.learnable_diagonal_quadratic import LearnableDiagonalQuadratic

class DeformedGaussianSolver(EncoderDecoderSolver):
    def __init__(self, forward_operator, diffeomorphism, lambd):
        super().__init__(diffeomorphism.d, forward_operator)
        self.phi = diffeomorphism
        self.psi = LearnableDiagonalQuadratic(self.d, use_softplus=True)
        self.lambd = lambd
    
    def reconstruction_loss(self,y):
        z = self.encode(y)
        z[:,-1] = 0.
        return 0.5 * ((self.decode_to_corrupted_data(z) - y)**2).mean(0).sum() + self.lambd * (self.psi.forward(z).mean() + 0.5 * self.psi.diagonal.log().sum())