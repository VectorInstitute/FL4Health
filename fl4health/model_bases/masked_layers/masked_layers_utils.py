import copy

from torch import nn

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
    Given a model, convert every one of its layers to a masked layer of the same kind, if applicable.

    Args:
        original_model (nn.Module): Module to be converted to a masked module.

    Returns:
        (nn.Module): New copy of the original model but with masked layers injected to enable FedPM.
    """

    def replace_with_masked(module: nn.Module) -> None:
        # Replace layers with their masked versions.
        for name, child in module.named_children():
            # Linear layers
            if isinstance(child, nn.Linear) and not isinstance(child, MaskedLinear):
                setattr(module, name, MaskedLinear.from_pretrained(child))
            # 1d, 2d, 3d convolutional layers and transposed convolutional layers
            elif isinstance(child, nn.Conv1d) and not isinstance(child, MaskedConv1d):
                setattr(module, name, MaskedConv1d.from_pretrained(child))
            elif isinstance(child, nn.Conv2d) and not isinstance(child, MaskedConv2d):
                setattr(module, name, MaskedConv2d.from_pretrained(child))
            elif isinstance(child, nn.Conv3d) and not isinstance(child, MaskedConv3d):
                setattr(module, name, MaskedConv3d.from_pretrained(child))
            elif isinstance(child, nn.ConvTranspose1d) and not isinstance(child, MaskedConvTranspose1d):
                setattr(module, name, MaskedConvTranspose1d.from_pretrained(child))
            elif isinstance(child, nn.ConvTranspose2d) and not isinstance(child, MaskedConvTranspose2d):
                setattr(module, name, MaskedConvTranspose2d.from_pretrained(child))
            elif isinstance(child, nn.ConvTranspose3d) and not isinstance(child, MaskedConvTranspose3d):
                setattr(module, name, MaskedConvTranspose3d.from_pretrained(child))
            # LayerNorm
            elif isinstance(child, nn.LayerNorm) and not isinstance(child, MaskedLayerNorm):
                setattr(module, name, MaskedLayerNorm.from_pretrained(child))
            # 1d, 2d, and 3d BatchNorm
            elif isinstance(child, nn.BatchNorm1d):
                setattr(module, name, MaskedBatchNorm1d.from_pretrained(child))
            elif isinstance(child, nn.BatchNorm2d):
                setattr(module, name, MaskedBatchNorm2d.from_pretrained(child))
            elif isinstance(child, nn.BatchNorm3d):
                setattr(module, name, MaskedBatchNorm3d.from_pretrained(child))
            # Recursively process the submodules of child
            else:
                replace_with_masked(child)

    # Deepcopy the model to avoid modifying the original
    masked_model = copy.deepcopy(original_model)
    replace_with_masked(masked_model)
    return masked_model


def is_masked_module(module: nn.Module) -> bool:
    """
    Checks whether the provided module is a masked module of the kind supported.

    Args:
        module (nn.Module): Module to be checked

    Returns:
        (bool): True if the module is a masked type and False otherwise.
    """
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
