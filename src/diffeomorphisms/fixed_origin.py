from src.diffeomorphisms import Diffeomorphism

class FixedOriginDiffeomorphism(Diffeomorphism):
    def __init__(self, diffeomorphism, origin):
        super().__init__(diffeomorphism.d)
        self.phi = diffeomorphism
        self.o = origin

    def forward(self, x, **kwargs):
        return self.phi.forward(x, **kwargs) - self.phi.forward(self.o[None], **kwargs)
    
    def inverse(self, y, **kwargs):
        return self.phi.inverse(y + self.phi.forward(self.o[None], **kwargs), **kwargs)
    
    def differential_forward(self, x, X, **kwargs):
        return self.phi.differential_forward(x, X, **kwargs)
    
    def differential_inverse(self, y, Y, **kwargs):
        return self.phi.differential_inverse(y + self.phi.forward(self.o[None], **kwargs), Y, **kwargs)