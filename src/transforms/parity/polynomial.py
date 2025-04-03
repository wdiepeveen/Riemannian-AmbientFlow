from src.transforms.parity import ParityTransform
from src.nn.module.activation.polynomial import PolynomialActivation

class PolynomialParityTransform(ParityTransform):
    def __init__(self, features, activation_class, order=2, parity=0):
        super().__init__(features, activation_class, PolynomialActivation, activation_args={'order':order}, parity=parity)
