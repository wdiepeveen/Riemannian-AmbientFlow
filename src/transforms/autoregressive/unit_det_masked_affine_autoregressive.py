import torch
from torch.nn import functional as F
from nflows.transforms import made as made_module
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.utils import torchutils

class UnitDetMaskedAffineAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.features = features
        made = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super().__init__(made)

    def _output_dim_multiplier(self):
        return 1

    def _elementwise_forward(self, inputs, autoregressive_params):
        shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        outputs = inputs + shift
        logabsdet = torch.zeros(inputs.shape[0])
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        shift = self._unconstrained_scale_and_shift(
            autoregressive_params
        )
        outputs = inputs - shift
        logabsdet = torch.zeros(inputs.shape[0])
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        autoregressive_params = autoregressive_params.view(
            -1, self.features, self._output_dim_multiplier()
        )
        # unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 0]
        return shift