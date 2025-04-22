from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Config, Parameters
from torch.nn import Module
import math


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


def get_granuarity(bit_width, k, model_dim, sigma, clipping_bound, n):
    t = 4 ** bit_width / 4 * math.pow(k, 2)
    print(t)
    gamma_square = 4 * (t/n - math.pow(clipping_bound, 2) * n / model_dim - math.pow(sigma, 2))
    print(math.sqrt(gamma_square))

if __name__ == '__main__':
    get_granuarity(bit_width=16, k=2, model_dim=1018174, sigma=9.5e-4, clipping_bound=0.03, n=3)
    get_granuarity(bit_width=16, k=2, model_dim=1018174, sigma=9.5e-4, clipping_bound=0.03, n=100)