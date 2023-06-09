from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithLayerNames
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


class NormDriftLayerExchanger(ParameterExchangerWithLayerNames):
    def __init__(self, initial_model: nn.Module, threshold: Scalar) -> None:
        """
        self.initial_model represents each client's local model at the beginning of each round of training.
        In this particular layer exchanger, self.initial_model is used to select
        the parameters that after local training drift away (in l2 norm) from
        the parameters of self.initial_model beyond a certain threshold.
        """
        self.initial_model = initial_model
        self.threshold = threshold

    def filter_layers(self, model: nn.Module) -> Tuple[NDArrays, List[str]]:
        """
        Return those layers of model that deviate (in l2 norm) away from corresponding layers of
        self.initial_model by at least self.threshold.
        """
        layer_names = []
        layers_to_transfer = []
        initial_model_states = self.initial_model.state_dict()
        model_states = model.state_dict()
        for layer_name in model_states:
            layer_param = model_states[layer_name]
            layer_param_past = initial_model_states[layer_name]
            drift_norm = torch.norm(layer_param - layer_param_past)
            if drift_norm >= self.threshold:
                layers_to_transfer.append(layer_param.cpu().numpy())
                layer_names.append(layer_name)
        return layers_to_transfer, layer_names

    def update_threshold(self, new_threshold: Scalar) -> None:
        self.threshold = new_threshold

    def push_parameters(self, model: nn.Module, config: Optional[Config] = None) -> NDArrays:
        layers_to_transfer, layer_names = self.filter_layers(model)
        return self.pack_parameters(layers_to_transfer, layer_names)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        # After updating each client model with the aggregated parameters sent by the server,
        # self.initial_model is also updated to the same parameters,
        # but it doesn't participate in the next round of training
        super().pull_parameters(parameters, model, config)
        self.initial_model.load_state_dict(model.state_dict())
