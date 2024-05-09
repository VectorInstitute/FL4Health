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
