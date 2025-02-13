import torch

from src.inverse_problem_solver import InverseProblemSolver

class AmbientFlowSolver(InverseProblemSolver):
    def __init__(self, forward_operator, prior_diffeomorphism, posterior_diffeomorphism, noise_level, sparsity_level, num_samples=10, reg_param=1.):
        super().__init__(prior_diffeomorphism.d, forward_operator)
        self.phi = prior_diffeomorphism
        self.phy = posterior_diffeomorphism
        self.sigma = noise_level
        self.k = sparsity_level
        self.M = num_samples
        self.mu = reg_param

    def reconstruction_loss(self, y):
        # Compute Phi
        # Phi = self.phi.differential_inverse(torch.zeros(self.d, self.d), torch.eye(self.d, self.d)).norm(2,-1)

        # Initialize loss components
        logavgexp_terms = []
        # sparsity_terms = []
        for _ in range(self.M):
            # Sample from p_eta(x|y)
            x_reconstructed = self.phy.nflow.sample(1, context=y).squeeze() 

            # Compute individual terms in the logavgexp expression
            log_p_theta = self.phi.nflow.log_prob(x_reconstructed) 
            log_q_n = -(1/(2 * self.sigma**2) * (self.A.forward(x_reconstructed) - y)**2).sum(dim=tuple(range(1,y.dim())))
            log_p_eta = self.phy.nflow.log_prob(x_reconstructed, context=y)

            logavgexp_terms.append(log_p_theta + log_q_n - log_p_eta)

            # # Compute individual terms in the sparsity expression
            # z_reconstructed = self.phi.forward(x_reconstructed)
            # Z_reconstructed = Phi * z_reconstructed
            # Z_projected = self.hard_thresholding(Z_reconstructed)

            # sparsity_terms.append(Z_reconstructed - Z_projected)


        # Compute logavgexp over all samples
        logavgexp_loss = (torch.logsumexp(torch.stack(logavgexp_terms), dim=0) - torch.log(torch.tensor(self.M, dtype=torch.float32).to(y.device))).mean()
        # sparsity_loss = torch.stack(sparsity_terms).abs().mean([0,1]).sum()

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