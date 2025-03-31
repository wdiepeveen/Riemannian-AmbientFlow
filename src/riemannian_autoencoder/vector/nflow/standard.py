import torch

from src.riemannian_autoencoder.vector.nflow import NFlowVectorRiemannianAutoencoder

class StandardNFlowVectorRiemannianAutoencoder(NFlowVectorRiemannianAutoencoder):
    def __init__(self, pullback_vector_euclidean, epsilon=None, d_epsilon=None):
        super().__init__(pullback_vector_euclidean)
        self.phi = self.manifold.phi

        # Compute covariance matrix
        D_0_phi_inv = self.phi.differential_inverse(torch.zeros(self.d, self.d), torch.eye(self.d, self.d))
        tangent_space_cov_matrix = D_0_phi_inv @ D_0_phi_inv.T
        Sigma, U = torch.linalg.eigh(tangent_space_cov_matrix)

        # Reverse the order to sort eigenvalues in descending order
        sorted_idx = torch.argsort(Sigma, descending=True)
        self.Sigma = Sigma[sorted_idx]
        self.U = U[:, sorted_idx]

        if epsilon is not None:
            assert d_epsilon is None
            if self.Sigma[-1] <= epsilon * self.Sigma.sum():
                # Use torch operations to avoid list comprehension issues
                cumulative_sum = self.Sigma.cumsum(dim=0)  # get cumulative sum
                remaining_sum = self.Sigma.sum() - cumulative_sum  # compute remaining sums
                tmp = remaining_sum <= epsilon * self.Sigma.sum()

                tmp_indices = torch.arange(0, self.d-1, device=self.Sigma.device)[tmp[:-1]]  # exclude last one from range
                if len(tmp_indices) > 0:
                    self.d_eps = tmp_indices.min() + 1
                else:
                    self.d_eps = self.d  # fallback case
                self.eps = self.Sigma[self.d_eps:].sum() / self.Sigma.sum()
            else:
                self.d_eps = self.d
                self.eps = 0.
        else:
            assert d_epsilon is not None
            self.d_eps = d_epsilon
            if self.d_eps < self.d:
                self.eps = self.Sigma[self.d_eps:].sum() / self.Sigma.sum()
            else:
                self.eps_effective = 0.

        print(f"constructed a Riemannian autoencoder with d_eps = {self.d_eps} and effectice eps = {self.eps}")

    def encode(self, x):
        x_bar = self.phi.inverse(torch.zeros(1,self.d,device=x.device))[0]
        log_x_bar_x = self.manifold.log(x_bar, x)
        return torch.einsum("Ni,ij->Nj", log_x_bar_x, self.U[:, :self.d_eps])
    
    def decode(self, p):
        x_bar = self.phi.inverse(torch.zeros(1,self.d,device=p.device))[0]
        Xi = torch.einsum("Nj,ij->Ni", p, self.U[:, :self.d_eps])
        print(Xi.shape)
        return self.manifold.exp(x_bar, Xi)
    