import torch

from src.inverse_problems import InverseProblem

class PnPRiemannianAmbientFlowProblem(InverseProblem):
    def __init__(self, forward_operator, prior_nflow_diffeomorphism, posterior_distribution, noise_level, num_samples=10, low_rank_reg_param=None):
        super().__init__(prior_nflow_diffeomorphism.d, forward_operator)
        self.phi_prior = prior_nflow_diffeomorphism
        self.p_post = posterior_distribution
        self.sigma = noise_level
        self.M = num_samples
        self.mu = low_rank_reg_param

    def reconstruction_loss(self, y, x_y):
        # Initialize loss components
        logavgexp_terms = []
        for _ in range(self.M):
            # Sample from p_post(x|y)
            x_reconstructed = self.p_post.sample(1, context=x_y)[:,0]

            # Compute individual terms in the logavgexp expression
            log_p_prior = self.phi_prior.nflow.log_prob(x_reconstructed) 
            log_p_noise = -(1/(2 * self.sigma**2) * (self.forward_operator.forward(x_reconstructed) - y)**2).sum(dim=tuple(range(1,y.dim())))
            log_p_post = self.p_post.log_prob(x_reconstructed, context=x_y)

            logavgexp_terms.append(log_p_prior + log_p_noise - log_p_post)


        # Compute logavgexp over all samples
        loss = -(torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()

        # Regularization
        if self.mu is not None:
            D_0_phi_inv = self.phi_prior.adjoint_differential_inverse(torch.zeros(self.d, self.d, device=y.device).reshape(self.d, *x_reconstructed.shape[1:]), 
                                                                    torch.eye(self.d, self.d, device=y.device).reshape(self.d, *x_reconstructed.shape[1:])
                                                                    ).reshape(self.d, self.d)
            loss += self.mu_1 * torch.linalg.norm(D_0_phi_inv, ord='fro')

        return loss
    