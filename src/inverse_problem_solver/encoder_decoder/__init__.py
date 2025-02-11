from src.inverse_problem_solver import InverseProblemSolver

class EncoderDecoderSolver(InverseProblemSolver):
    def __init__(self, forward_operator, diffeomorphism, sparsity_level):
        super().__init__(diffeomorphism.d, forward_operator)
        self.phi = diffeomorphism
        self.k = sparsity_level

    def reconstruct(self, y):
        return self.decode_to_data(self.encode(y))

    def decode_to_corrupted_data(self, z):
        return self.A.forward(self.phi.inverse(z))
    
    def decode_to_data(self, z):
        return self.phi.inverse(z)
    
    def encode(self, y): 
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def reconstruction_loss(self,y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def k_sparse_project(self, z):
        return 5 # TODO