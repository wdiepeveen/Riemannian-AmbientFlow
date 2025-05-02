import torch.nn as nn

class InverseProblem(nn.Module):
    def __init__(self, d, forward_operator):
        super().__init__()
        self.d = d
        self.forward_operator = forward_operator
    
    def reconstruction_loss(self, y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )