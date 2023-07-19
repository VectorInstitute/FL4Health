import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames


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

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        return self.apply_layer_filter(model)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters
        for layer_name, layer_parameters in zip(self.layers_to_transfer, parameters):
            current_state[layer_name] = torch.tensor(layer_parameters)
        model.load_state_dict(current_state, strict=True)


class NormDriftLayerExchanger(ParameterExchangerWithPacking[List[str]]):
    def __init__(self, threshold: Scalar, exchange_percentage: float = 0.1) -> None:
        """
        self.initial_model represents each client's local model at the beginning of each round of training.
        In this particular layer exchanger, self.initial_model is used to select
        the parameters that after local training drift away (in l2 norm) from
        the parameters of self.initial_model beyond a certain threshold.
        """
        self.parameter_packer = ParameterPackerWithLayerNames()
        self.threshold = threshold
        self.exchange_percentage = exchange_percentage

    def filter_layers(self, model: nn.Module, initial_model: nn.Module) -> Tuple[NDArrays, List[str]]:
        """
        Return those layers of model that deviate (in l2 norm) away from corresponding layers of
        self.initial_model by at least self.threshold.
        """
        layer_names = []
        layers_to_transfer = []
        initial_model_states = initial_model.state_dict()
        model_states = model.state_dict()
        for layer_name in model_states:
            layer_param = model_states[layer_name]
            layer_param_past = initial_model_states[layer_name]
            drift_norm = torch.norm((layer_param - layer_param_past).float())
            if drift_norm >= self.threshold:
                layers_to_transfer.append(layer_param.cpu().numpy())
                layer_names.append(layer_name)
        return layers_to_transfer, layer_names

    def set_threshold(self, new_threshold: Scalar) -> None:
        self.threshold = new_threshold

    def set_percentage(self, new_percentage: float) -> None:
        assert 0 < new_percentage <= 1
        self.exchange_percentage = new_percentage

    def filter_layers_by_percentage(self, model: nn.Module, initial_model: nn.Module) -> Tuple[NDArrays, List[str]]:
        names_to_norm_drift = {}
        initial_model_states = initial_model.state_dict()
        model_states = model.state_dict()

        for layer_name, layer_param in model_states.items():
            layer_param_past = initial_model_states[layer_name]
            drift_norm = torch.norm((layer_param - layer_param_past).float())
            names_to_norm_drift[layer_name] = drift_norm

        total_param_num = len(names_to_norm_drift.keys())
        num_param_exchange = int(math.ceil(total_param_num * self.exchange_percentage))
        param_to_exchange_names = sorted(names_to_norm_drift.keys(), key=lambda x: names_to_norm_drift[x])[
            : (num_param_exchange + 1)
        ]
        print(param_to_exchange_names)
        return [model_states[name].cpu().numpy() for name in param_to_exchange_names], param_to_exchange_names

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        assert initial_model is not None
        layers_to_transfer, layer_names = self.filter_layers_by_percentage(model, initial_model)
        return self.pack_parameters(layers_to_transfer, layer_names)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters
        layer_params, layer_names = self.unpack_parameters(parameters)
        for layer_name, layer_param in zip(layer_names, layer_params):
            current_state[layer_name] = torch.tensor(layer_param)
        model.load_state_dict(current_state, strict=True)
