from logging import WARNING
from typing import Tuple

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters
from opacus.validators import ModuleValidator


def get_all_model_parameters(model: nn.Module) -> Parameters:
    # Extracting all model parameters and converting to Parameters object
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])


def privacy_validate_and_fix_modules(model: nn.Module) -> Tuple[nn.Module, bool]:
    # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
    # with compatible ones if necessary.

    errors = ModuleValidator.validate(model, strict=False)
    reinitialize_optimizer = len(errors) > 0
    # Due to a bug in Opacus, it's possible that we need to run multiple rounds fo module validator fix for
    # complex nested models to fully replace all layers within a model (for example, in the Fed-IXI model)
    while len(errors) != 0:
        for error in errors:
            opacus_warning = (
                "Opacus has found layers within your model, that do not comply with DP training. "
                "These layers will automatically be replaced with DP compliant layers. "
                "If you would like to perform this replacement yourself, please adjust your model manually."
            )
            log(WARNING, f"{opacus_warning}")
            log(WARNING, f"Opacus error: {error}")
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
    # If we made changes to the underlying model, we may need to reinitialize an optimizer
    return model, reinitialize_optimizer
