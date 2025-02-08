import torch

from src.dimension_reduction.principal_geodesic_analysis.vector import l2PGAVectorSolver
from src.dimension_reduction.principal_geodesic_analysis.vector.l2_tangent_space_pca import l2TangentSpacePCAVectorSolver

class l2CurvatureCorrectedPCAVectorSolver(l2PGAVectorSolver):
    def __init__(self, data, vector_euclidean, base_point, h=1e-6) -> None:
        super().__init__(data, vector_euclidean, base_point)

        self.l2PGA_initialisation = l2TangentSpacePCAVectorSolver(self.data, self.euclidean, self.base_point)

        self.h = h
        self.beta_x_data = None # ∈ R^{n x d x d}

        self.U = {i+1: None for i in range(self.d)} 
        self.Sigma = {i+1: None for i in range(self.d)}
        self.V = {i+1: None for i in range(self.d)}

        # TODO construct a setter to update the l2PGA_initialisation

    def compress(self, rank):
        assert rank > 0
        if self.Xi[rank] is None:
            if self.log_x_data is None:
                # compute log
                self.log_x_data = self.euclidean.log(self.base_point, self.data)  # ∈ R^{n x d}
            
            if self.l2PGA_initialisation.log_x_data is None:
                self.l2PGA_initialisation.log_x_data = self.log_x_data

            if self.beta_x_data is None:
                # compute n x d differentials D_log_x_data_i exp_x [e^j]
                D_exp_x = torch.zeros(self.N, self.d, self.d, device=self.base_point.device) # ∈ R^{n x d x d}
                for i in range(self.N):
                        if i%10 == 0:
                            print(f"i = {i}")
                        exp_x_minus = self.euclidean.exp(self.base_point, self.log_x_data[i][None] - self.h * torch.eye(self.d, device=self.base_point.device))
                        exp_x_plus = self.euclidean.exp(self.base_point, self.log_x_data[i][None] + self.h * torch.eye(self.d, device=self.base_point.device))
                        D_exp_x[i] = 1 / (2 * self.h) * (exp_x_plus - exp_x_minus) 
                
                # compute betas from differentials
                self.beta_x_data = torch.einsum("ijm,ikm->ijk", D_exp_x, D_exp_x) # ∈ R^{n x d x d}

            # tangent space SVD
            self.l2PGA_initialisation.solve(rank)

            # curvature correction step (compute W)
            # compute matrix A ∈ R^{d x r x d x r}
            A = torch.einsum("mij,mk,ml->ikjl", self.beta_x_data, self.l2PGA_initialisation.U[rank], self.l2PGA_initialisation.U[rank])
            # compute vector b ∈ R^{d x r}
            b = torch.einsum("lij,lk,lj->ik", self.beta_x_data, self.l2PGA_initialisation.U[rank], self.log_x_data)
            # solve linear system to retrieve W
            W = torch.linalg.solve(A.reshape(self.d * rank, self.d * rank), b.reshape(self.d * rank)).reshape(self.d, rank)
            
            # find U, Sigma and V from SVD of QW to get orthonormal vectors again
            self.Xi[rank] = torch.einsum("ik,jk->ij", self.l2PGA_initialisation.U[rank], W)

            U, Sigma, V = torch.svd(self.Xi[rank])

            self.U[rank] = U[:,0:rank]
            self.Sigma[rank] =  Sigma[0:rank]
            self.V[rank] = V[:,0:rank]
        else:
            print(f"rank-{rank} already computed")