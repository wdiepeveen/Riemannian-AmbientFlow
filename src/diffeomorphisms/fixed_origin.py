from src.diffeomorphisms import Diffeomorphism

class FixedOriginDiffeomorphism(Diffeomorphism):
    def __init__(self, diffeomorphism, origin):
        super().__init__(diffeomorphism.d)
        self.phi = diffeomorphism
        self.o = origin

    def forward(self, x):
        return self.phi.forward(x) - self.phi.forward(self.o[None])
    
    def inverse(self, y):
        return self.phi.inverse(y + self.phi.forward(self.o[None]))
    
    def differential_forward(self, x, X):
        return 5 # TODO
    
    def differential_inverse(self, y, Y):
        return 6 # TODO