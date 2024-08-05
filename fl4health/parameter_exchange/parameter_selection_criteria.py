import math
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import NDArray, NDArrays
from scipy.stats import bernoulli
from torch import Tensor

from fl4health.model_bases.masked_layers import is_masked_module
from fl4health.utils.typing import LayerSelectionFunction


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


# Helper functions for select_scores_and_sample_masks
def _sample_masks(score_tensor: Tensor) -> NDArray:
    bernoulli_probabilities = torch.sigmoid(score_tensor).cpu().numpy()
    # Perform Bernoulli sampling.
    binary_mask = bernoulli.rvs(bernoulli_probabilities)
    return binary_mask


def _process_masked_module(
    module: nn.Module, model_state_dict: Dict[str, Tensor], module_name: Optional[str] = None
) -> Tuple[NDArrays, List[str]]:
    """
    Perform Bernoulli sampling using the weight and bias scores of a masked module.

    Args:
        module (nn.Module): the module upon which operations described above are performed.
            "module" can either be a submodule of the model trained in FedPM, or it can a standalone module itself.
            In the latter case, the argument "model_state_dict" should be the same as "module.state_dict()".
            In either case, it is assumed that module is a masked module.
        model_state_dict (Dict[str, Tensor]): the state dictionary of the model trained in FedPM.
        module_name (Optional[str]): the name of module if module is a submodule of the model trained in FedPM.
            This is used to access the weight and bias score tensors in model_state_dict. Defaults to None.
    """
    masks_to_exchange = []
    score_tensor_names = []
    # If module_name is passed in, then we prepend it to "weight_scores" to get the correct
    # key in the state dictionary.
    weight_scores_tensor_name = f"{module_name}.weight_scores" if module_name else "weight_scores"
    score_tensor_names.append(weight_scores_tensor_name)
    weight_scores = model_state_dict[weight_scores_tensor_name]

    # Note: due to the Bernoulli sampling performed here, the parameters selected are in fact binary masks
    # even though their corresponding names are something like "weight_scores" or "bias_scores".
    # After the tensors have been aggregated by the strategy, they will become score tensors again.
    # This misalignment was allowed because these parameter names will later be used to load the model anyway.
    masks_to_exchange.append(_sample_masks(weight_scores))
    # Do the same thing with bias_scores if it exists
    if "bias_scores" in module.state_dict().keys():
        bias_scores_tensor_name = f"{module_name}.bias_scores" if module_name else "bias_scores"
        score_tensor_names.append(bias_scores_tensor_name)
        bias_scores = model_state_dict[bias_scores_tensor_name]
        masks_to_exchange.append(_sample_masks(bias_scores))
    return masks_to_exchange, score_tensor_names


def select_scores_and_sample_masks(model: nn.Module, initial_model: Optional[nn.Module]) -> Tuple[NDArrays, List[str]]:
    """
    Selection function that first selects the "weight_scores" and "bias_scores" parameters for the
    masked layers, and then samples binary masks based on those scores to send to the server.
    This function is meant to be used for the FedPM algorithm.

    Note: in the current implementation, we always exchange the score tensors for all layers. In the future, we might
        support exchanging a subset of the layers (for example, filtering out the masks that are all zeros).
    """
    model_states = model.state_dict()
    with torch.no_grad():
        if is_masked_module(model):
            return _process_masked_module(module=model, model_state_dict=model_states)
        else:
            masks_to_exchange = []
            score_tensor_names = []
            for name, module in model.named_modules():
                if is_masked_module(module):
                    module_masks, module_score_tensor_names = _process_masked_module(
                        module=module, model_state_dict=model_states, module_name=name
                    )
                    masks_to_exchange.extend(module_masks)
                    score_tensor_names.extend(module_score_tensor_names)
            return masks_to_exchange, score_tensor_names
