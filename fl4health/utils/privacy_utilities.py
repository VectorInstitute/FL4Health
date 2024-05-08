from logging import INFO, WARNING
from typing import Any, Tuple

import torch.nn as nn
from flwr.common.logger import log
from opacus import GradSampleModule
from opacus.grad_sample.utils import wrap_model
from opacus.validators import ModuleValidator


def privacy_validate_and_fix_modules(model: nn.Module) -> Tuple[nn.Module, bool]:
    """
    This function runs Opacus model validation to ensure that the provided models layers are compatible with the
    privacy mechanisms in Opacus. The function attempts to use Opacus to replace any incompatible layers if possible.
    For example BatchNormalization layers are not "DP Compliant" and will be replaced by compliant layers such as
    GroupNormalization with this function. Note that this uses the default "fix" functionality in Opacus. For more
    custom options, defining your own setup_opacus_objects function is required.

    Args:
        model (nn.Module): The model to be validated and potentially modified to be Opacus compliant.

    Returns:
        Tuple[nn.Module, bool]: Returns a (possibly) modified pytorch model and a boolean indicating whether a
            reinitialization of any optimizers associated with the model will be required. Reinitialization of the
            optimizer parameters is required, for example, when the model layers are modified, yielding a mismatch
            in the optimizer parameters and the new model parameters.
    """
    errors = ModuleValidator.validate(model, strict=False)
    reinitialize_optimizer = len(errors) > 0
    # Due to a bug in Opacus, it's possible that we need to run multiple rounds fo module validator fix for
    # complex nested models to fully replace all layers within a model (for example, in the Fed-IXI model)
    while len(errors) != 0:
        for error in errors:
            opacus_warning = (
                "Opacus has found layers within your model that do not comply with DP training. "
                "These layers will automatically be replaced with DP compliant layers. "
                "If you would like to perform this replacement yourself, please adjust your model manually."
            )
            log(WARNING, f"{opacus_warning}")
            log(WARNING, f"Opacus error: {error}")
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
    # If we made changes to the underlying model, we may need to reinitialize an optimizer
    return model, reinitialize_optimizer


def convert_model_to_opacus_model(
    model: nn.Module, grad_sample_mode: str = "hooks", *args: Any, **kwargs: Any
) -> GradSampleModule:
    """
    This function converts a standard pytorch model to an Opacus GradSampleModule, which Opacus uses to perform
    efficient DP-SGD operations. It uses the wrap_model functionality and mimics its defaults.

    Args:
        model (nn.Module): Pytorch model to be converted to an Opacus GradSampleModule
        grad_sample_mode (str, optional): This determines how Opacus performs the conversion under the hood. The
            standard mechanism is indicated by "hooks" but other approaches may be necessary depending on how the
            pytorch module is defined. Defaults to "hooks".

    Returns:
        GradSampleModule: The Opacus wrapped GradSampleModule
    """
    if isinstance(model, GradSampleModule):
        log(INFO, f"Provided model is already of type {type(model)}, skipping conversion to Opacus model type")
        return model
    return wrap_model(model, grad_sample_mode=grad_sample_mode, *args, **kwargs)


def map_model_to_opacus_model(
    model: nn.Module, grad_sample_mode: str = "hooks", *args: Any, **kwargs: Any
) -> GradSampleModule:
    """
    Performs an validation and modifications necessary to make the provided pytorch model "Opacus Compliant" via the
    call to privacy_validate_and_fix_modules. The resulting model is then converted to an Opacus GradSampleModule via
    convert_model_to_opacus_model.

    Args:
        model (nn.Module): Pytorch model to be converted to an Opacus compliant GradSampleModule
        grad_sample_mode (str, optional): This determines how Opacus performs the conversion under the hood. The
            standard mechanism is indicated by "hooks" but other approaches may be necessary depending on how the
            pytorch module is defined. Defaults to "hooks".

    Returns:
        GradSampleModule: The Opacus-compliant, wrapped GradSampleModule
    """
    model, _ = privacy_validate_and_fix_modules(model)
    return convert_model_to_opacus_model(model, grad_sample_mode, *args, **kwargs)
