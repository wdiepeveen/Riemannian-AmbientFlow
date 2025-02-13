from src.diffeomorphisms.vector.nflow import NFlowVectorDiffeomorphism

class FixedOriginNFlowVectorDiffeomorphism(NFlowVectorDiffeomorphism):
    def __init__(self, nflow_vector_diffeomorphism, origin):
        super().__init__(nflow_vector_diffeomorphism.d, nflow_vector_diffeomorphism.nflow)
        self.o = origin

    def forward(self, x, context=None):
        return super().forward(x, context=context) - super().forward(self.o[None], context=context)
    
    def inverse(self, y, context=None):
        return super().inverse(y + super().forward(self.o[None], context=context), context=context)
    
    def differential_forward(self, x, X, context=None):
        return super().differential_forward(x, X, context=context)
    
    def differential_inverse(self, y, Y, context=None):
        return super().differential_inverse(y + super().forward(self.o[None], context=context), Y, context=context)