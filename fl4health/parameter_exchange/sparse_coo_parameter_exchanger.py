import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays
from torch import Tensor
from torch.nn.modules import Module

from fl4health.parameter_exchange.parameter_packer import SparseCooParameterPacker
from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger


class SparseCooParameterExchanger(PartialParameterExchanger[Tuple[NDArrays, NDArrays, List[str]]]):
    def __init__(self, sparsity_level: float, score_gen_function: Callable[..., Dict[str, Tensor]]) -> None:
        """
        Parameter exchanger for sparse tensors.

        This exchanger is responsible for selecting an arbitrary subset of a model's parameters
        via some selection criterion and then packaging them into the COO sparse tensor format for exchanging.

        Args:
            sparsity_level (float): The level of sparsity. Must be between 0 and 1.
            score_gen_function (Callable[..., Dict[str, Tensor]]): Function that is responsible for
            generating a score for every parameter inside a model in order to facilitate parameter selection.

            In most cases, this function takes as inputs a current model and an initial model,
            and it returns a dictionary that maps the names of a model's tensors to another tensor
            which contains the parameter scores.
        """
        assert 0 < sparsity_level <= 1
        self.set_sparsity_level(sparsity_level)
        self.parameter_packer: SparseCooParameterPacker = SparseCooParameterPacker()
        self.set_score_gen_function(score_gen_function)

    def set_sparsity_level(self, sparsity_level: float) -> None:
        """This method exists to allow for setting new sparsity levels throughout training."""
        self.sparsity_level = sparsity_level

    def set_score_gen_function(self, score_gen_function: Callable[..., Dict[str, Tensor]]) -> None:
        self.score_gen_function = score_gen_function

    def generate_parameter_scores(self, model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
        """Calling the score generating function to produce parameter scores."""
        return self.score_gen_function(model, initial_model)

    def create_masks(self, model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
        """
        Produce masks for model's tensors.

        More precisely, this method first leverage a score generating function
        to generate scores for all parameters of model.

        Next, these scores are used to produce masks.
        A threshold is determined according to the desired sparsity level,
        then parameters whose scores are less than this threshold are assigned a mask value of 0,
        while parameters whose scores are greater than this threshold are assigned a mask value of 1.

        Args:
            model (nn.Module): Current model.
            initial_model (nn.Module): Initial model.

        Returns:
            Dict[str, Tensor]: Dictionary that maps the name of each tensor to its corresponding mask.
        """
        all_parameter_scores = self.generate_parameter_scores(model, initial_model)
        all_scores = torch.cat([val.flatten() for _, val in all_parameter_scores.items()])
        # Sorting all scores and determining the threshold.
        sorted_scores, _ = torch.sort(all_scores, descending=True)
        top_scores = sorted_scores[: math.ceil(len(sorted_scores) * self.sparsity_level) + 1]
        score_threshold = top_scores[-1].item()

        names_to_masks = {}
        for name, param_scores in all_parameter_scores.items():
            # Use score_threshold to produce masks.
            threshold_result = torch.threshold(param_scores, threshold=score_threshold, value=0)
            mask = torch.where(threshold_result == 0, input=threshold_result, other=1)
            # Tensor whose mask values are all zero will not be exchanged, so we
            # do not return them.
            if not (mask == 0).all():
                names_to_masks[name] = mask
        return names_to_masks

    def select_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None
    ) -> Tuple[NDArrays, Tuple[NDArrays, NDArrays, List[str]]]:
        tensor_names_to_masks = self.create_masks(model, initial_model)
        model_states = model.state_dict()

        selected_parameters_all_tensors = []
        selected_indices_all_tensors = []
        tensor_shapes = []
        tensor_names = list(tensor_names_to_masks.keys())

        # Note that param_names_to_masks only contains those tensors whose
        # mask values are not all zero.
        for tensor_name, tensor_mask in tensor_names_to_masks.items():
            model_tensor = model_states[tensor_name]
            assert model_tensor.shape == tensor_mask.shape
            sparse_tensor = torch.multiply(model_tensor, tensor_mask)
            selected_parameters, selected_indices, tensor_shape = self.parameter_packer.extract_coo_info_from_dense(
                sparse_tensor
            )

            selected_parameters_all_tensors.append(selected_parameters)
            selected_indices_all_tensors.append(selected_indices)
            tensor_shapes.append(tensor_shape)

        return (selected_parameters_all_tensors, (selected_indices_all_tensors, tensor_shapes, tensor_names))

    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        selected_parameters, additional_parameters = self.select_parameters(model, initial_model)
        return self.pack_parameters(
            model_weights=selected_parameters,
            additional_parameters=additional_parameters,
        )

    def pull_parameters(self, parameters: NDArrays, model: Module, config: Optional[Config] = None) -> None:
        selected_parameters, additional_info = self.parameter_packer.unpack_parameters(parameters)
        indices, shapes, names = additional_info
        current_state = model.state_dict()

        for param_values, param_indices, param_shape, param_name in zip(selected_parameters, indices, shapes, names):
            # Use parameter values, indices, and shape to create
            # a sparse coo tensor, which is then converted to a dense tensor
            # to allow for loading.
            param_coo = torch.sparse_coo_tensor(
                indices=torch.tensor(param_indices.T), values=torch.tensor(param_values), size=torch.Size(param_shape)
            )
            param_dense = param_coo.to_dense()
            current_state[param_name] = param_dense

        model.load_state_dict(current_state, strict=True)
