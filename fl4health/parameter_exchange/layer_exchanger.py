import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

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
    def __init__(
        self,
        norm_threshold: float,
        exchange_percentage: float = 0.1,
        normalize: bool = True,
        filter_by_percentage: bool = True,
    ) -> None:
        """
        self.initial_model represents each client's local model at the beginning of each round of training.
        In this particular layer exchanger, self.initial_model is used to select
        the parameters that after local training drift away (in l2 norm) from
        the parameters of self.initial_model beyond a certain threshold.
        """
        self.parameter_packer = ParameterPackerWithLayerNames()
        assert 0 < exchange_percentage <= 1
        assert 0 < norm_threshold
        self.threshold = norm_threshold
        self.exchange_percentage = exchange_percentage
        self.set_normalization_mode(normalize)
        self.set_filter_mode(filter_by_percentage)

    def set_filter_mode(self, filter_by_percentage: bool) -> None:
        self.filter_by_percentage = filter_by_percentage

    def set_normalization_mode(self, normalize: bool) -> None:
        self.normalize = normalize

    def _calculate_drift_norm(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        t_diff = (t1 - t2).float()
        drift_norm = torch.linalg.norm(t_diff)
        if self.normalize:
            drift_norm /= len(torch.flatten(t_diff))
        return drift_norm.item()

    def filter_layers_by_threshold(self, model: nn.Module, initial_model: nn.Module) -> Tuple[NDArrays, List[str]]:
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
            drift_norm = self._calculate_drift_norm(layer_param, layer_param_past)
            if drift_norm >= self.threshold:
                layers_to_transfer.append(layer_param.cpu().numpy())
                layer_names.append(layer_name)
        return layers_to_transfer, layer_names

    def set_threshold(self, new_threshold: float) -> None:
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
            drift_norm = self._calculate_drift_norm(layer_param, layer_param_past)
            names_to_norm_drift[layer_name] = drift_norm

        total_param_num = len(names_to_norm_drift.keys())
        num_param_exchange = int(math.ceil(total_param_num * self.exchange_percentage))
        param_to_exchange_names = sorted(names_to_norm_drift.keys(), key=lambda x: names_to_norm_drift[x])[
            : (num_param_exchange + 1)
        ]
        return [model_states[name].cpu().numpy() for name in param_to_exchange_names], param_to_exchange_names

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        assert initial_model is not None
        if self.filter_by_percentage:
            layers_to_transfer, layer_names = self.filter_layers_by_percentage(model, initial_model)
        else:
            layers_to_transfer, layer_names = self.filter_layers_by_threshold(model, initial_model)
        return self.pack_parameters(layers_to_transfer, layer_names)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters
        layer_params, layer_names = self.unpack_parameters(parameters)
        for layer_name, layer_param in zip(layer_names, layer_params):
            current_state[layer_name] = torch.tensor(layer_param)
        model.load_state_dict(current_state, strict=True)
