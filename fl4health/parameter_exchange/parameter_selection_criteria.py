import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays
from torch import Tensor


# Selection criteria functions for selecting entire layers. Should be used with the DynamicLayerExchanger class.
def _calculate_drift_norm(t1: torch.Tensor, t2: torch.Tensor, normalize: bool) -> float:
    t_diff = (t1 - t2).float()
    drift_norm = torch.linalg.norm(t_diff)
    if normalize:
        drift_norm /= torch.numel(t_diff)
    return drift_norm.item()


def select_layers_by_threshold(
    model: nn.Module, initial_model: nn.Module, threshold: float, normalize: bool, select_drift_more: bool
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
        layer_param_past = initial_model_states[layer_name]
        drift_norm = _calculate_drift_norm(layer_param, layer_param_past, normalize)
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
    model: nn.Module, initial_model: nn.Module, exchange_percentage: float, normalize: bool, select_drift_more: bool
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


# Selection criteria functions for selecting arbitrary sets of weights. Related to the super-mask paper.
def largest_final_magnitude_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    for tensor_name, tensor_values in model.state_dict().items():
        names_to_scores[tensor_name] = torch.abs(tensor_values)
    return names_to_scores


def smallest_final_magnitude_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    for tensor_name, tensor_values in model.state_dict().items():
        names_to_scores[tensor_name] = (-1) * torch.abs(tensor_values)
    return names_to_scores


def largest_magnitude_drift_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = torch.abs(current_tensor_values - initial_tensor_values)
    return names_to_scores


def smallest_magnitude_drift_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = (-1) * torch.abs(current_tensor_values - initial_tensor_values)
    return names_to_scores


def largest_change_in_magnitude_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = torch.abs(torch.abs(current_tensor_values) - torch.abs(initial_tensor_values))
    return names_to_scores


def smallest_change_in_magnitude_scores(model: nn.Module, initial_model: nn.Module) -> Dict[str, Tensor]:
    names_to_scores = {}
    current_model_states = model.state_dict()
    initial_model_states = initial_model.state_dict()
    for tensor_name, current_tensor_values in current_model_states.items():
        initial_tensor_values = initial_model_states[tensor_name]
        names_to_scores[tensor_name] = (-1) * torch.abs(
            torch.abs(current_tensor_values) - torch.abs(initial_tensor_values)
        )
    return names_to_scores
