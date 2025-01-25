from src.dimension_reduction import DimensionReduction

class l2PGASolver(DimensionReduction):
    """ Implements base class for solving the PGA problem of finding some rank-r matrix Xi_x \in R^{n x d} such that \|Y - exp_x(Xi_x)\|_F^2 is small """
    def __init__(self, data, euclidean_manifold, base_point) -> None:
        super().__init__(data)

        self.euclidean_manifold = euclidean_manifold
        self.base_point = base_point

        self.log_x_data = None
        self.Xi = {} # dictionary with elements in R^{n x d}

    def solve(self, rank):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def evaluate(self, rank):
        # compute the loss as is from the Xi obtained -- ideally also store this in another dictionary
        return 5