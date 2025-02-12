import torch

from src.inverse_problem_solver import InverseProblemSolver

class AmbientFlowSolver(InverseProblemSolver):
    def __init__(self, forward_operator, prior_diffeomorphism, posterior_diffeomorphism, sparsity_level, num_samples=10, reg_param=1.):
        super().__init__(prior_diffeomorphism.d, forward_operator)
        self.phi = prior_diffeomorphism
        self.phy = posterior_diffeomorphism
        self.k = sparsity_level
        self.M = num_samples
        self.mu = reg_param

    def reconstruction_loss(self, y):
        # Sample from standard normal distribution for latent variables
        zeta_samples = torch.randn(self.M, *y.shape).to(y.device)  # Shape: (M, batch_size, d)

        # Compute Phi
        Phi = self.phi.differential_inverse(torch.zeros(self.d, self.d), torch.eye(self.d, self.d)).norm(2,-1)

        # Initialize loss components
        logavgexp_terms = []
        sparsity_terms = []
        for i in range(self.M):
            # Compute the inverse mapping of the conditional diffeomorphism
            x_reconstructed = self.phy.inverse(zeta_samples[i], context=y)  # phi_eta^{-1}(zeta_i; y)

            # Compute individual terms in the logavgexp expression
            log_p_theta = self.phi.nflow.log_prob(x_reconstructed)  # log p_theta(phi_eta^{-1}(zeta_i; y))
            log_q_n = (1/2 * (y - self.A.forward(x_reconstructed))**2).sum(dim=tuple(range(1,y.dim())))  # log q_n(y - Aphi_eta^{-1}(zeta_i; y))
            log_p_eta = self.phy.nflow.log_prob(x_reconstructed, context=y)  # log p_eta(phi_eta^{-1}(zeta_i; y) | y)

            logavgexp_terms.append(log_p_theta + log_q_n - log_p_eta)

            # Compute individual terms in the sparsity expression
            z_reconstructed = self.phi.forward(x_reconstructed)
            Z_reconstructed = Phi * z_reconstructed
            Z_projected = self.hard_thresholding(Z_reconstructed)

            sparsity_terms.append(Z_reconstructed - Z_projected)


        # Compute logavgexp over all samples
        logavgexp_loss = (torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()
        sparsity_loss = torch.stack(sparsity_terms).abs().mean([0,1]).sum()

        # Combine terms with weighting
        total_loss = -logavgexp_loss # + self.mu * sparsity_loss

        return total_loss
    
    def hard_thresholding(self, x):
        # Get the top-k values and their indices along the last dimension
        topk_values, topk_indices = torch.topk(torch.abs(x), k=self.k, dim=1, largest=True, sorted=False)

        # Create a mask for the top-k elements
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # Apply the mask to retain only the top-k elements
        thresholded_tensor = torch.where(mask, x, torch.zeros_like(x))

        return thresholded_tensor