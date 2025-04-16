from src.transforms.multi_layer_parity_conv2d import MultiLayerParityConv2DTransform
from src.nn.module.activation.sine import SineActivation

class MultiLayerSineParityConv2DTransform(MultiLayerParityConv2DTransform):
    def __init__(self, in_channels, height, width, kernel_size, latent_channels, parity=0):
        super().__init__(in_channels, height, width, kernel_size, latent_channels, SineActivation, activation_args={}, parity=parity)
        