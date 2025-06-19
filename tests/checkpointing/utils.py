from opacus import GradSampleModule
from torch import nn

from fl4health.utils.privacy_utilities import convert_model_to_opacus_model


def create_opacus_model_via_functorch(model: nn.Module) -> GradSampleModule:
    return convert_model_to_opacus_model(model, grad_sample_mode="functorch")
