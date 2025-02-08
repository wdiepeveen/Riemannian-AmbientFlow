import torch.nn as nn

class InverseProblemSolver(nn.Module):
    def __init__(self, d, forward_operator):
        super().__init__()
        self.d = d
        self.A = forward_operator

    def reconstruct(self, y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
    
    def reconstruction_loss(self,y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )