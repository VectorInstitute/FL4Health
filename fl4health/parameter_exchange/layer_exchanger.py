from typing import List, Optional

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger


class FixedLayerExchanger(ParameterExchanger):
    def __init__(self, layers_to_transfer: List[str]) -> None:
        self.layers_to_transfer = layers_to_transfer

    def apply_layer_filter(self, model: nn.Module) -> NDArrays:
        # NOTE: Filtering layers only works if each client exchanges exactly the same layers
        return [
            layer_parameters.cpu().numpy()
            for layer_name, layer_parameters in model.state_dict().items()
            if layer_name in self.layers_to_transfer
        ]

    def push_parameters(self, model: nn.Module, config: Optional[Config] = None) -> NDArrays:
        return self.apply_layer_filter(model)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters
        for layer_name, layer_parameters in zip(self.layers_to_transfer, parameters):
            current_state[layer_name] = torch.tensor(layer_parameters)
        model.load_state_dict(current_state, strict=True)
