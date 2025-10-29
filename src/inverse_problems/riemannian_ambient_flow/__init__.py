import torch

from src.inverse_problems import InverseProblem

class RiemannianAmbientFlowProblem(InverseProblem):
    def __init__(self, forward_operator, prior_nflow_diffeomorphism, posterior_distribution, noise_level, num_samples=10, low_rank_reg_param=None, clean_data_reg_param=None):
        super().__init__(prior_nflow_diffeomorphism.d, forward_operator)
        self.phi_prior = prior_nflow_diffeomorphism
        self.p_post = posterior_distribution
        self.sigma = noise_level
        self.M = num_samples
        self.mu_1 = low_rank_reg_param
        self.mu_2 = clean_data_reg_param

    def reconstruction_loss(self, y, x=None):
        # Initialize loss components
        logavgexp_terms = []
        for _ in range(self.M):
            # Sample from p_post(x|y)
            x_reconstructed = self.p_post.sample(1, context=y)[:,0]

            # Compute individual terms in the logavgexp expression
            log_p_prior = self.phi_prior.nflow.log_prob(x_reconstructed) 
            log_p_noise = -(1/(2 * self.sigma**2) * (self.forward_operator.forward(x_reconstructed) - y)**2).sum(dim=tuple(range(1,y.dim())))
            log_p_post = self.p_post.log_prob(x_reconstructed, context=y)

            logavgexp_terms.append(log_p_prior + log_p_noise - log_p_post)


        # Compute logavgexp over all samples
        loss = -(torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()

        # Regularization
        if self.mu_1 is not None:
            D_0_phi_inv = self.phi_prior.adjoint_differential_inverse(torch.zeros(self.d, self.d, device=y.device).reshape(self.d, *x_reconstructed.shape[1:]), 
                                                                    torch.eye(self.d, self.d, device=y.device).reshape(self.d, *x_reconstructed.shape[1:])
                                                                    ).reshape(self.d, self.d)
            loss += self.mu_1 * torch.linalg.norm(D_0_phi_inv, ord='fro')

        if self.mu_2 is not None and x is not None:
            log_p_prior_gt = self.phi_prior.nflow.log_prob(x)
            loss -= self.mu_2 * log_p_prior_gt.mean()

        return loss
    
    # def get_context(self, y):
    #     x_init = self.forward_operator.pseudo_inverse(y)
    #     return torch.cat([x_init, torch.zeros_like(x_init)], dim=1)
    
    # def reconstruct(self, y):
    #     return self.p_post._context_encoder(self.get_context(y)).chunk(2, dim=-1)[0]
    