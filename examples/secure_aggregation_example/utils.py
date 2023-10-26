from torch.nn import Module
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters

# the following function is consumed by the server strategy 
def generate_config(local_epochs: int, batch_size: int, current_server_round: int) -> Config:

    package = {
                "local_epochs": local_epochs, 
                "batch_size": batch_size, 
                "current_server_round": current_server_round,
    }

    return package

# the following function is consumed by the server strategy 
def get_parameters(model: Module) -> Parameters:
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
