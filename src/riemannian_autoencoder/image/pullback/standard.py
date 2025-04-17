import torch

from src.riemannian_autoencoder.image.pullback import PullbackImageRiemannianAutoencoder

class StandardPullbackImageRiemannianAutoencoder(PullbackImageRiemannianAutoencoder):
    def __init__(self, pullback_image_euclidean, epsilon=None, d_epsilon=None):
        super().__init__(pullback_image_euclidean)
        self.data_shape = pullback_image_euclidean.data_shape
        self.phi = self.manifold.phi

        # Compute covariance matrix
        D_0_phi_inv_T = self.phi.adjoint_differential_inverse(torch.zeros(self.d, self.d).reshape(self.d, self.C, self.H, self.W), torch.eye(self.d, self.d).reshape(self.d, self.C, self.H, self.W)).reshape(self.d, self.d)
        tangent_space_cov_matrix = D_0_phi_inv_T.T @ D_0_phi_inv_T
        Sigma, U = torch.linalg.eigh(tangent_space_cov_matrix)

        # Reverse the order to sort eigenvalues in descending order
        sorted_idx = torch.argsort(Sigma, descending=True)
        self.Sigma = Sigma[sorted_idx]
        self.U = U[:, sorted_idx]
        for i in range(self.d):
            print(f"{i} | sigma = {self.Sigma[i]} with u = {self.U[:,i]}")

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
        x_bar = self.phi.inverse(torch.zeros(1, self.C, self.W, self.H, device=x.device))[0]
        log_x_bar_x = self.manifold.log(x_bar, x)
        return torch.einsum("Ni,ij->Nj", log_x_bar_x.reshape(-1, self.d), self.U[:, :self.d_eps])
    
    def decode(self, p):
        x_bar = self.phi.inverse(torch.zeros(1, self.C, self.W, self.H, device=p.device))[0]
        Xi = torch.einsum("Nj,ij->Ni", p, self.U[:, :self.d_eps]).reshape(-1, self.C, self.H, self.W)
        return self.manifold.exp(x_bar, Xi)
    