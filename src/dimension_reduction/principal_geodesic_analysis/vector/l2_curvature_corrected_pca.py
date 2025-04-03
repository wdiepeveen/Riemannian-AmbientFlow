import torch

from src.dimension_reduction.principal_geodesic_analysis.vector import l2PGAVectorSolver
from src.dimension_reduction.principal_geodesic_analysis.vector.l2_tangent_space_pca import l2TangentSpacePCAVectorSolver

class l2CurvatureCorrectedPCAVectorSolver(l2PGAVectorSolver):
    def __init__(self, data, vector_euclidean, base_point, device="cpu", h=1e-6) -> None:
        super().__init__(data, vector_euclidean, base_point, device)

        self.l2PGA_initialisation = l2TangentSpacePCAVectorSolver(self.data, self.euclidean, self.base_point)

        self.h = h
        self.beta_x_data = self.get_beta_x_data()

    def get_Xi(self, rank):
        assert rank > 0
        # curvature correction step (compute W)
        # compute matrix A ∈ R^{d x r x d x r}
        A = torch.einsum("mij,mk,ml->ikjl", self.beta_x_data, self.l2PGA_initialisation.U[:,0:rank], self.l2PGA_initialisation.U[:,0:rank])
        # compute vector b ∈ R^{d x r}
        b = torch.einsum("lij,lk,lj->ik", self.beta_x_data, self.l2PGA_initialisation.U[:,0:rank], self.log_x_data)
        # solve linear system to retrieve W
        W = torch.linalg.solve(A.reshape(self.d * rank, self.d * rank), b.reshape(self.d * rank)).reshape(self.d, rank)
        
        return torch.einsum("ik,jk->ij", self.l2PGA_initialisation.U[:,0:rank], W)
    
    def get_beta_x_data(self):
        # compute n x d differentials D_log_x_data_i exp_x [e^j]
        D_exp_x = torch.zeros(self.N, self.d, self.d) # ∈ R^{n x d x d}
        for i in range(self.N):
                if i%10 == 0:
                    print(f"i = {i}")
                exp_x_minus = self.euclidean.exp(self.base_point.to(self.device), self.log_x_data[i][None].to(self.device) - self.h * torch.eye(self.d, device=self.device)).detach().cpu()
                exp_x_plus = self.euclidean.exp(self.base_point.to(self.device), self.log_x_data[i][None].to(self.device) + self.h * torch.eye(self.d, device=self.device)).detach().cpu()
                D_exp_x[i] = 1 / (2 * self.h) * (exp_x_plus - exp_x_minus) 
        
        # compute betas from differentials
        return torch.einsum("ijm,ikm->ijk", D_exp_x, D_exp_x) # ∈ R^{n x d x d}