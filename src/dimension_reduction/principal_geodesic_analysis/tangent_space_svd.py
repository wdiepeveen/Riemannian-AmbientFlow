import torch

from src.dimension_reduction.principal_geodesic_analysis import l2PGASolver

class TangentSpacel2SVD(l2PGASolver):
    def __init__(self, data, euclidean_manifold, base_point) -> None:
        super().__init__(data, euclidean_manifold, base_point)

        self.U = {} 
        self.Sigma = {}
        self.V = {}

    def solve(self, rank):
        assert rank > 0 and rank <= min(self.data_dim, self.data_size)
        if rank not in self.Xi:
            if self.log_x_data is None:
                # compute log
                self.log_x_data = self.euclidean_manifold.log(self.base_point, self.data)  # ∈ R^{n x d}

            # compute svd
            U, Sigma, V = torch.svd(self.log_x_data.cpu())

            self.U[rank] = U[:,0:rank].to(self.base_point.device)
            self.Sigma[rank] =  Sigma[0:rank].to(self.base_point.device)
            self.V[rank] = V[:,0:rank].to(self.base_point.device)

            self.Xi[rank] = torch.einsum("ij,j,kj->ik", self.U[rank], self.Sigma[rank], self.V[rank])
        else:
            print(f"rank-{rank} already computed")
