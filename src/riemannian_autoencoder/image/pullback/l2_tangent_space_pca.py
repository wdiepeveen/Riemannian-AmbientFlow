import torch

from src.dimension_reduction.principal_geodesic_analysis.image.l2_tangent_space_pca import l2TangentSpacePCAImageSolver
from src.riemannian_autoencoder.image.pullback import PullbackImageRiemannianAutoencoder

class l2TangentSpacePCAPullbackImageRiemannianAutoencoder(PullbackImageRiemannianAutoencoder):
    def __init__(self, pullback_image_euclidean, data, latent_dim):
        super().__init__(pullback_image_euclidean)
        self.data = data
        self.N = self.data.shape[0]
        self.base_point = self.manifold.barycentre(self.data)

        self.dim_red = l2TangentSpacePCAImageSolver(self.data, self.manifold, self.base_point)
        self.update_dim(latent_dim)

        for i in range(self.d):
            print(f"{i} | sigma = {self.dim_red.Sigma[i]} with u = {self.dim_red.V[:,i]}")

        if self.m < self.d:
                self.eps = self.dim_red.Sigma[self.m:].sum() / self.dim_red.Sigma.sum()
        else:
            self.eps_effective = 0.
        print(f"constructed a Riemannian autoencoder with latent dimension = {self.m} and effectice eps = {self.eps}")

    def update_dim(self, dim):
        assert dim > 0 and dim <= min(self.d, self.N)
        self.m = dim
        self.Sigma = self.dim_red.Sigma[0:dim]
        self.V = self.dim_red.V[:,0:dim]

    def encode(self, x):
        log_q_x = self.manifold.log(self.base_point.to(x.device), x).reshape(-1, self.d)
        return torch.einsum("Ni,ij->Nj", log_q_x, self.V)
    
    def decode(self, p):
        assert p.shape[1] == self.m
        Xi = torch.einsum("Nj,ij->Ni", p, self.V.to(p.device)).reshape(-1, self.C, self.H, self.W)
        return self.manifold.exp(self.base_point.to(p.device), Xi)

        
