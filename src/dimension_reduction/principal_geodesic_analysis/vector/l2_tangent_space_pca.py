import torch

from src.dimension_reduction.principal_geodesic_analysis.vector import l2PGAVectorSolver

class l2TangentSpacePCAVectorSolver(l2PGAVectorSolver):
    def __init__(self, data, vector_euclidean, base_point) -> None:
        super().__init__(data, vector_euclidean, base_point)

        self.U = {i+1: None for i in range(self.d)} 
        self.Sigma = {i+1: None for i in range(self.d)}
        self.V = {i+1: None for i in range(self.d)}

    def compress(self, rank):
        assert rank > 0 and rank <= min(self.d, self.N)
        if self.Xi[rank] is None:
            if self.log_x_data is None:
                # compute log
                self.log_x_data = self.euclidean.log(self.base_point, self.data)  # ∈ R^{n x d}

            # compute svd
            U, Sigma, V = torch.svd(self.log_x_data.cpu())

            self.U[rank] = U[:,0:rank].to(self.base_point.device)
            self.Sigma[rank] =  Sigma[0:rank].to(self.base_point.device)
            self.V[rank] = V[:,0:rank].to(self.base_point.device)

            self.Xi[rank] = torch.einsum("ij,j,kj->ik", self.U[rank], self.Sigma[rank], self.V[rank])
        else:
            print(f"rank-{rank} already computed")
