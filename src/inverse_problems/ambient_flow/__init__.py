import torch

from src.inverse_problems import InverseProblem

class AmbientFlowProblem(InverseProblem):
    def __init__(self, forward_operator, prior_diffeomorphism, posterior_diffeomorphism, noise_level, num_samples=10, reg_param=None):
        super().__init__(prior_diffeomorphism.d, forward_operator)
        self.phi = prior_diffeomorphism
        self.phy = posterior_diffeomorphism
        self.sigma = noise_level
        self.M = num_samples
        self.mu = reg_param

    def reconstruct(self, y): # TODO
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def reconstruction_loss(self, y):
        # Initialize loss components
        logavgexp_terms = []
        for _ in range(self.M):
            # Sample from p_eta(x|y)
            x_reconstructed = self.phy.nflow.sample(1, context=y).squeeze() 

            # Compute individual terms in the logavgexp expression
            log_p_theta = self.phi.nflow.log_prob(x_reconstructed) 
            log_q_n = -(1/(2 * self.sigma**2) * (self.A.forward(x_reconstructed) - y)**2).sum(dim=tuple(range(1,y.dim())))
            log_p_eta = self.phy.nflow.log_prob(x_reconstructed, context=y)

            logavgexp_terms.append(log_p_theta + log_q_n - log_p_eta)


        # Compute logavgexp over all samples
        loss = -(torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()

        # Regularization
        if self.mu is not None:
            D_0_phi_inv = self.phi.adjoint_differential_inverse(torch.zeros(self.d, self.d), torch.eye(self.d, self.d))
            loss += self.mu * torch.linalg.norm(D_0_phi_inv, ord='fro')

        return loss
    