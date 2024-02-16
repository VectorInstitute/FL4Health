import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays
from torch import Tensor

LayerSelectionFunction = Callable[[nn.Module, nn.Module], Tuple[NDArrays, List[str]]]


class LayerSelectionFunctionConstructor:
    def __init__(
        self, norm_threshold: float, exchange_percentage: float, normalize: bool = True, select_drift_more: bool = True
    ) -> None:
        """
        This class leverages functools.partial to construct layer selection functions,
        which are meant to be used by the DynamicLayerExchanger class.

        Args:
            norm_threshold (float): A nonnegative real number used to select those
                layers whose drift in l2 norm exceeds (or falls short of) it.
            exchange_percentage (float): Indicates the percentage of layers that are selected.
            normalize (bool, optional): Indicates whether when calculating the norm of a layer,
                we also divide by the number of parameters in that layer. Defaults to True.
            select_drift_more (bool, optional): Indicates whether layers with larger
                drift norm are selected. Defaults to True.
        """
        assert 0 < exchange_percentage <= 1
        assert 0 < norm_threshold
        self.norm_threshold = norm_threshold
        self.exchange_percentage = exchange_percentage
        self.normalize = normalize
        self.select_drift_more = select_drift_more

    def select_by_threshold(self) -> LayerSelectionFunction:
        return partial(
            select_layers_by_threshold,
            self.norm_threshold,
            self.normalize,
            self.select_drift_more,
        )

    def select_by_percentage(self) -> LayerSelectionFunction:
        return partial(
            select_layers_by_percentage,
            self.exchange_percentage,
            self.normalize,
            self.select_drift_more,
        )


# Selection criteria functions for selecting entire layers. Intended to be used
# by the DynamicLayerExchanger class via the LayerSelectionFunctionConstructor class.
def _calculate_drift_norm(t1: torch.Tensor, t2: torch.Tensor, normalize: bool) -> float:
    t_diff = (t1 - t2).float()
    drift_norm = torch.linalg.norm(t_diff)
    if normalize:
        drift_norm /= torch.numel(t_diff)
    return drift_norm.item()


def select_layers_by_threshold(
    threshold: float,
    normalize: bool,
    select_drift_more: bool,
    model: nn.Module,
    initial_model: nn.Module,
) -> Tuple[NDArrays, List[str]]:
    """
    Return those layers of model that deviate (in l2 norm) away from corresponding layers of
    self.initial_model by at least (or at most) self.threshold.
    """
    layer_names = []
    layers_to_transfer = []
    initial_model_states = initial_model.state_dict()
    model_states = model.state_dict()
    for layer_name, layer_param in model_states.items():
        ghost_of_layer_params_past = initial_model_states[layer_name]
        drift_norm = _calculate_drift_norm(layer_param, ghost_of_layer_params_past, normalize)
        if select_drift_more:
            if drift_norm > threshold:
                layers_to_transfer.append(layer_param.cpu().numpy())
                layer_names.append(layer_name)
        else:
            if drift_norm <= threshold:
                layers_to_transfer.append(layer_param.cpu().numpy())
                layer_names.append(layer_name)
    return layers_to_transfer, layer_names


def select_layers_by_percentage(
    exchange_percentage: float,
    normalize: bool,
    select_drift_more: bool,
    model: nn.Module,
    initial_model: nn.Module,
) -> Tuple[NDArrays, List[str]]:
    names_to_norm_drift = {}
    initial_model_states = initial_model.state_dict()
    model_states = model.state_dict()

    for layer_name, layer_param in model_states.items():
        layer_param_past = initial_model_states[layer_name]
        drift_norm = _calculate_drift_norm(layer_param, layer_param_past, normalize)
        names_to_norm_drift[layer_name] = drift_norm

    total_param_num = len(names_to_norm_drift.keys())
    num_param_exchange = int(math.ceil(total_param_num * exchange_percentage))
    param_to_exchange_names = sorted(
        names_to_norm_drift.keys(), key=lambda x: names_to_norm_drift[x], reverse=select_drift_more
    )[:(num_param_exchange)]
    return [model_states[name].cpu().numpy() for name in param_to_exchange_names], param_to_exchange_names


# Score generating functions used for selecting arbitrary sets of weights.
# The ones implemented here are those that demonstrated good performance in the super-mask paper.
# Link to this paper: https://arxiv.org/abs/1905.01067
def largest_final_magnitude_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    names_to_scores = {}
    for tensor_name, tensor_values in model.state_dict().items():
        names_to_scores[tensor_name] = torch.abs(tensor_values)
    return names_to_scores


def smallest_final_magnitude_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    names_to_scores = {}
    for tensor_name, tensor_values in model.state_dict().items():
        names_to_scores[tensor_name] = (-1) * torch.abs(tensor_values)
    return names_to_scores


def largest_magnitude_change_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    assert initial_model is not None
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = torch.abs(current_tensor_values - initial_tensor_values)
    return names_to_scores


def smallest_magnitude_change_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    assert initial_model is not None
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = (-1) * torch.abs(current_tensor_values - initial_tensor_values)
    return names_to_scores


def largest_increase_in_magnitude_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    assert initial_model is not None
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = torch.abs(current_tensor_values) - torch.abs(initial_tensor_values)
    return names_to_scores


def smallest_increase_in_magnitude_scores(model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
    assert initial_model is not None
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = (-1) * (torch.abs(current_tensor_values) - torch.abs(initial_tensor_values))
    return names_to_scores
