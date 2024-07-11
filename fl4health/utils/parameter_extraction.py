from typing import Iterable

import torch
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters


def get_all_model_parameters(model: nn.Module) -> Parameters:
    """
    Function to extract ALL parameters associated with a pytorch module, including any state parameters. These
    values are converted from numpy arrays into a Flower Parameters object.

    Args:
        model (nn.Module): PyTorch model whose parameters are to be extracted

    Returns:
        Parameters: Flower Parameters object containing all of the target models state.
    """
    # Extracting all model parameters and converting to Parameters object
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])


def check_shape_match(params1: Iterable[torch.Tensor], params2: Iterable[torch.Tensor], error_message: str) -> None:
    """
    Check if the shapes of parameters from two models match.

    Args:
        params1 (Iterable[torch.Tensor]): Parameters from the first model.
        params2 (Iterable[torch.Tensor]): Parameters from the second model.
        error_message (str): Error message to display if the shapes do not match.
    """
    params1_list = list(params1)
    params2_list = list(params2)

    # Check if the number of parameters match
    assert len(params1_list) == len(
        params2_list
    ), f"Parameter length mismatch: \
        {len(params1_list)} vs {len(params2_list)}. {error_message}"

    # Check if each corresponding parameter shape matches
    for param1, param2 in zip(params1_list, params2_list):
        assert param1.shape == param2.shape, error_message
