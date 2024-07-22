from typing import Optional

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger
from fl4health.utils.functions import sigmoid_inverse


class FedPmExchanger(DynamicLayerExchanger):
    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        layer_params, layer_names = self.unpack_parameters(parameters)
        for layer_name, layer_param in zip(layer_names, layer_params):
            # Apply the inverse of the Sigmoid function
            # since the scores for masked layers are supposed to be unbounded.
            with torch.no_grad():
                current_state[layer_name] = sigmoid_inverse(torch.tensor(layer_param))
        model.load_state_dict(current_state, strict=True)
