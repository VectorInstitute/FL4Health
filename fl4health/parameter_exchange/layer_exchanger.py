import math
from typing import List, Optional, Set, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames

T = TypeVar("T", bound=nn.Module)


class FixedLayerExchanger(ParameterExchanger):
    def __init__(self, layers_to_transfer: List[str]) -> None:
        self.layers_to_transfer = layers_to_transfer

    def apply_layer_filter(self, model: nn.Module) -> NDArrays:
        # NOTE: Filtering layers only works if each client exchanges exactly the same layers
        model_state_dict = model.state_dict()
        return [model_state_dict[layer_to_transfer].cpu().numpy() for layer_to_transfer in self.layers_to_transfer]

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


class LayerExchangerWithExclusions(ParameterExchanger):
    """
    This class implements exchanging all model layers except those matching a specified set of types. The constructor
    is provided with model in order to extract the proper layers to be exchanged based on the exclusion criteria
    """

    def __init__(self, model: nn.Module, module_exclusions: Set[Type[T]]) -> None:
        self.module_exclusions = module_exclusions
        self.modules_to_filter: Set[str] = {
            # Note: Remove duplicate needs to be false in case modules have been tied together with shared objects.
            name
            for name, module in model.named_modules(remove_duplicate=False)
            if self.should_module_be_excluded(module) and name
        }
        # Needs to be an ordered collection to facilitate exchange consistency between server and client
        self.layers_to_transfer: List[str] = self.get_layers_to_transfer(model)

    def should_module_be_excluded(self, module: Type[T]) -> bool:
        return type(module) in self.module_exclusions

    def should_layer_be_excluded(self, layer_name: str) -> bool:
        return any([layer_name.startswith(module_to_filter) for module_to_filter in self.modules_to_filter])

    def get_layers_to_transfer(self, model: nn.Module) -> List[str]:
        return [name for name in model.state_dict().keys() if not self.should_layer_be_excluded(name)]

    def apply_layer_filter(self, model: nn.Module) -> NDArrays:
        # NOTE: Filtering layers only works if each client exchanges exactly the same layers
        model_state_dict = model.state_dict()
        return [model_state_dict[layer_to_transfer].cpu().numpy() for layer_to_transfer in self.layers_to_transfer]

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


class NormDriftParameterExchanger(ParameterExchangerWithPacking[List[str]]):
    def __init__(
        self,
        norm_threshold: float,
        exchange_percentage: float = 0.1,
        normalize: bool = True,
        filter_by_percentage: bool = True,
    ) -> None:
        """
        This exchanger selects those parameters that at the end of each training round, drift away (in l2 norm)
        from their initial values (at the beginning of the same round) by more than self.threshold.
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
            drift_norm /= torch.numel(t_diff)
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
        for layer_name, layer_param in model_states.items():
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
            :(num_param_exchange)
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
