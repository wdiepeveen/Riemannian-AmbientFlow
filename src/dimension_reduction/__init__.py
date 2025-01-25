class DimensionReduction:
    """ Base class for dimension reduction of d-dimensional data Y = (Y_1, ..., Y_n)^T \in R^{n x d} """
    def __init__(self, data) -> None:
        assert len(data.shape) == 2
        self.data = data
        
        size, dim = data.shape
        self.data_dim = dim
        self.data_size = size

    def solve(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )