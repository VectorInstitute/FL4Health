from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters
from peft import get_peft_model_state_dict
from torch import nn


def get_all_peft_parameters_from_model(model: nn.Module) -> Parameters:
    """
    Function to extract peft parameters associated with a pytorch module. These values are converted
    from numpy arrays into a Flower Parameters object.

    Args:
        model (nn.Module): PyTorch model whose parameters are to be extracted.

    Returns:
        (Parameters): Flower Parameters object containing all of the target models state.
    """
    state_dict = get_peft_model_state_dict(model)
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in state_dict.items()])
