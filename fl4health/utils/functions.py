import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters


def get_all_model_parameters(model: nn.Module) -> Parameters:
    # Extracting all model parameters and converting to Parameters object
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
