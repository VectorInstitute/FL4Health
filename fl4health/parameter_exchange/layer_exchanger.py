from typing import Callable, List, Optional, Set, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames
from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger

TorchModule = TypeVar("TorchModule", bound=nn.Module)
LayerSelectionFunction = Callable[[nn.Module, nn.Module], Tuple[NDArrays, List[str]]]


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
    is provided with the model in order to extract the proper layers to be exchanged based on the exclusion criteria
    """

    def __init__(self, model: nn.Module, module_exclusions: Set[Type[TorchModule]]) -> None:
        # module_exclusion is a set of nn.Module types that should NOT be exchanged with the server.
        # {nn.BatchNorm1d}
        self.module_exclusions = module_exclusions
        # In order to filter out all weights associated with a module, we run through all of the named modules and
        # store the names of those that have a type matching the provided exclusions
        self.modules_to_filter: Set[str] = {
            # Note: Remove duplicate needs to be false in case modules have been tied together with shared objects.
            name
            for name, module in model.named_modules(remove_duplicate=False)
            # We store only if the module should be exclude and name is not the empty string
            if self.should_module_be_excluded(module) and name
        }
        # Needs to be an ordered collection to facilitate exchange consistency between server and client
        # NOTE: Layers here refers to a collection of parameters in the state dictionary
        self.layers_to_transfer: List[str] = self.get_layers_to_transfer(model)

    def should_module_be_excluded(self, module: Type[TorchModule]) -> bool:
        return type(module) in self.module_exclusions

    def should_layer_be_excluded(self, layer_name: str) -> bool:
        # The model state_dict prefixes the weights and/or state associated with a named module with the name of that
        # module and then an identifier for the specific parameters.
        # Ex. named module: name: "fc1" module: nn.Linear(10, 10, bias=True)
        # The state_dict has keys fc1.weight and fc1.bias with associated parameters
        # We filter out any parameters prefixed with the name of an excluded module, as stored in modules_to_filter
        return any([layer_name.startswith(module_to_filter) for module_to_filter in self.modules_to_filter])

    def get_layers_to_transfer(self, model: nn.Module) -> List[str]:
        # We store the state dictionary keys that do not correspond to excluded modules as held in modules_to_filter
        return [name for name in model.state_dict().keys() if not self.should_layer_be_excluded(name)]

    def apply_layer_filter(self, model: nn.Module) -> NDArrays:
        # NOTE: Filtering layers only works if each client exchanges exactly the same layers
        model_state_dict = model.state_dict()
        # The order of the parameters is determined by the order of layers to transfer, this ensures that they
        # always have the same order, which can be relied upon in weight reconstruction done by pull_parameters
        return [model_state_dict[layer_to_transfer].cpu().numpy() for layer_to_transfer in self.layers_to_transfer]

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        return self.apply_layer_filter(model)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters. Assumes order of parameters is the same as in push_parameters
        for layer_name, layer_parameters in zip(self.layers_to_transfer, parameters):
            current_state[layer_name] = torch.tensor(layer_parameters)
        model.load_state_dict(current_state, strict=True)


class DynamicLayerExchanger(PartialParameterExchanger[List[str]]):
    def __init__(
        self,
        layer_selection_function: LayerSelectionFunction,
    ) -> None:
        """
        This exchanger uses "layer_selection_function" to select a subset of a model's layers
        at the end of each training round. Only the selected layers are exchanged with the server.
        Args:
            layer_selection_function (LayerSelectionFunction):
                Function responsible for selecting the layers to be exchanged. This function relies
                on extra parameters such as norm threshold or exchange percentage,
                but we assume that it has already been pre-constructed using
                the class LayerSelectionFunctionConstructor, so it only needs to take
                in two nn.Module objects as inputs. For more details, please see the
                docstring of LayerSelectionFunctionConstructor.
        """
        self.layer_selection_function = layer_selection_function
        self.parameter_packer = ParameterPackerWithLayerNames()

    def select_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None
    ) -> Tuple[NDArrays, List[str]]:
        assert initial_model is not None
        return self.layer_selection_function(model, initial_model)

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        assert initial_model is not None
        layers_to_transfer, layer_names = self.select_parameters(model, initial_model)
        return self.pack_parameters(layers_to_transfer, layer_names)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        current_state = model.state_dict()
        # update the correct layers to new parameters
        layer_params, layer_names = self.unpack_parameters(parameters)
        for layer_name, layer_param in zip(layer_names, layer_params):
            current_state[layer_name] = torch.tensor(layer_param)
        model.load_state_dict(current_state, strict=True)
