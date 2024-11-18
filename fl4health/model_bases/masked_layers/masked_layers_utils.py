import copy

import torch.nn as nn

from fl4health.model_bases.masked_layers.masked_conv import (
    MaskedConv1d,
    MaskedConv2d,
    MaskedConv3d,
    MaskedConvTranspose1d,
    MaskedConvTranspose2d,
    MaskedConvTranspose3d,
)
from fl4health.model_bases.masked_layers.masked_linear import MaskedLinear
from fl4health.model_bases.masked_layers.masked_normalization_layers import (
    MaskedBatchNorm1d,
    MaskedBatchNorm2d,
    MaskedBatchNorm3d,
    MaskedLayerNorm,
    _MaskedBatchNorm,
)


def convert_to_masked_model(original_model: nn.Module) -> nn.Module:
    """
    Given a model, convert every one of its linear or convolutional layer to a masked layer
    of the same kind, as defined in the classes above.
    """
    masked_model = copy.deepcopy(original_model)
    for name, module in original_model.named_modules():
        # Mask nn.Linear modules
        if isinstance(module, nn.Linear) and not isinstance(module, MaskedLinear):
            setattr(masked_model, name, MaskedLinear.from_pretrained(module))

        # Mask convolutional modules (1d, 2d, and 3d)
        elif isinstance(module, nn.Conv1d) and not isinstance(module, MaskedConv1d):
            setattr(masked_model, name, MaskedConv1d.from_pretrained(module))
        elif isinstance(module, nn.Conv2d) and not isinstance(module, MaskedConv2d):
            setattr(masked_model, name, MaskedConv2d.from_pretrained(module))
        elif isinstance(module, nn.Conv3d) and not isinstance(module, MaskedConv3d):
            setattr(masked_model, name, MaskedConv3d.from_pretrained(module))

        # Mask transposed convolutional modules (1d, 2d, 3d)
        elif isinstance(module, nn.ConvTranspose1d) and not isinstance(module, MaskedConvTranspose1d):
            setattr(masked_model, name, MaskedConvTranspose1d.from_pretrained(module))
        elif isinstance(module, nn.ConvTranspose2d) and not isinstance(module, MaskedConvTranspose2d):
            setattr(masked_model, name, MaskedConvTranspose2d.from_pretrained(module))
        elif isinstance(module, nn.ConvTranspose3d) and not isinstance(module, MaskedConvTranspose3d):
            setattr(masked_model, name, MaskedConvTranspose3d.from_pretrained(module))

        # Mask nn.LayerNorm module
        elif isinstance(module, nn.LayerNorm) and not isinstance(module, MaskedLayerNorm):
            setattr(masked_model, name, MaskedLayerNorm.from_pretrained(module))

        # Mask batch norm modules (1d, 2d, and 3d)
        elif isinstance(module, nn.BatchNorm1d):
            setattr(masked_model, name, MaskedBatchNorm1d.from_pretrained(module))
        elif isinstance(module, nn.BatchNorm2d):
            setattr(masked_model, name, MaskedBatchNorm2d.from_pretrained(module))
        elif isinstance(module, nn.BatchNorm3d):
            setattr(masked_model, name, MaskedBatchNorm3d.from_pretrained(module))

    return masked_model


def is_masked_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            MaskedLinear,
            MaskedConv1d,
            MaskedConv2d,
            MaskedConv3d,
            MaskedConvTranspose1d,
            MaskedConvTranspose2d,
            MaskedConvTranspose3d,
            MaskedLayerNorm,
            _MaskedBatchNorm,
        ),
    )
