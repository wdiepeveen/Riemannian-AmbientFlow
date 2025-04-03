from src.dimension_reduction import DimensionReductionSolver

class l2PGASolver(DimensionReductionSolver):
    """ Implements base class for solving the PGA problem of finding some rank-r matrix Xi_x \in R^{n x d_1 x ... x d_n} such that \|Y - exp_x(Xi_x)\|_F^2 is small """
    def __init__(self, N, d, data, euclidean, base_point) -> None:
        super().__init__(N, d, data)
        self.data_norm = (self.data**2).mean(0).sum()

        self.euclidean = euclidean
        self.base_point = base_point

        self.log_x_data = None
        self.Xi = {i+1: None for i in range(self.d)}
        self.exp_x_Xi = {i+1: None for i in range(self.d)}
        self.error = {i+1: None for i in range(self.d)}
        self.rel_error = {i+1: None for i in range(self.d)}

    def solve(self, rank):
        print(f"Computing rank {rank} approximation on tangent space")
        self.compress(rank)
        print(f"Computing rank {rank} approximation on euclidean space")
        self.reconstruct(rank)
        print(f"Computing rank {rank} errors")
        self.evaluate_error(rank)
    
    def compress(self, rank):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def reconstruct(self, rank):
        if self.exp_x_Xi[rank] is None:
            self.exp_x_Xi[rank] = self.euclidean.exp(self.base_point, self.Xi[rank])
        else:
            print(f"rank-{rank} already computed")
    
    def evaluate_error(self, rank):
        if self.error[rank] is None:
            error = ((self.data - self.exp_x_Xi[rank])**2).mean(0).sum()
            self.error[rank] = error
            self.rel_error[rank] = error / self.data_norm
        else:
            print(f"rank-{rank} already computed")