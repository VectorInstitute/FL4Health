import math
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays
from torch import Tensor
from torch.nn.modules import Module

from fl4health.parameter_exchange.parameter_packer import SparseCooParameterPacker
from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger

ScoreGenFunction = Callable[[nn.Module, Optional[nn.Module]], Dict[str, Tensor]]


class SparseCooParameterExchanger(PartialParameterExchanger[Tuple[NDArrays, NDArrays, List[str]]]):
    def __init__(self, sparsity_level: float, score_gen_function: ScoreGenFunction) -> None:
        """
        Parameter exchanger for sparse tensors.

        This exchanger is responsible for selecting an arbitrary subset of a model's parameters
        via some selection criterion and then packaging them into the COO sparse tensor format for exchanging.

        For more information on the sparse COO format and sparse tensors in PyTorch, please see the following
        two pages:
            1. https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
            2. https://pytorch.org/docs/stable/sparse.html

        Args:
            sparsity_level (float): The level of sparsity. Must be between 0 and 1.
            score_gen_function (ScoreGenFunction): Function that is responsible for
            generating a score for every parameter inside a model in order to facilitate parameter selection.

            In most cases, this function takes as inputs a current model and an initial model,
            and it returns a dictionary that maps the name of each of the current model's tensors to
            another tensor which contains the parameter scores.
        """
        assert 0 < sparsity_level <= 1
        self.sparsity_level = sparsity_level
        self.parameter_packer: SparseCooParameterPacker = SparseCooParameterPacker()
        self.score_gen_function = score_gen_function

    def generate_parameter_scores(self, model: nn.Module, initial_model: Optional[nn.Module]) -> Dict[str, Tensor]:
        """Calling the score generating function to produce parameter scores."""
        return self.score_gen_function(model, initial_model)

    def _check_unique_score(self, param_scores: Tensor) -> None:
        unique_score_values = torch.unique(input=param_scores, sorted=False, return_inverse=False, return_counts=False)
        if len(unique_score_values) == 1:
            log(
                WARNING,
                """All parameters have the same score.
                The number of parameters selected may not match the intended sparsity level.""",
            )

    def select_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None
    ) -> Tuple[NDArrays, Tuple[NDArrays, NDArrays, List[str]]]:
        """
        Select model parameters according to the sparsity level and pack them into
        the sparse COO format to be exchanged.

        First, this method leverages a score generating function
        to generate scores for all parameters of model.

        Next, these scores are used to select the parameters to be exchanged by
        performing a thresholding operation on each of the model's tensors.
        A threshold is determined according to the desired sparsity level,
        then for each model tensor, parameters whose scores are less than this threshold
        are set to zero, while parameters whose scores are greater than or equal to
        this threshold retain their values.

        Finally, the method extracts all the information required to represent
        the selected parameters in the sparse COO tensor format. More specifically,
        the information consists of the indices of the parameters within the tensor
        to which they belong, the shape of that tensor, and also the name of it.

        Args:
            model (nn.Module): Current model.
            initial_model (nn.Module): Initial model.

        Returns:
            Tuple[NDArrays, Tuple[NDArrays, NDArrays, List[str]]]: the selected parameters
            and other information, as detailed above.
        """
        all_parameter_scores = self.generate_parameter_scores(model, initial_model)
        all_scores = torch.cat([val.flatten() for _, val in all_parameter_scores.items()])
        # Sorting all scores and determining the threshold.
        sorted_scores, _ = torch.sort(all_scores, descending=True)
        n_top_scores = math.ceil(len(sorted_scores) * self.sparsity_level)
        # Sanity check.
        assert n_top_scores >= 1
        score_threshold = sorted_scores[(n_top_scores - 1)].item()

        # Apply the score threshold to each model tensor to obtain the corresponding sparse tensor.
        selected_parameters_all_tensors = []
        selected_indices_all_tensors = []
        tensor_shapes = []
        tensor_names = []
        model_states = model.state_dict()
        for tensor_name, param_scores in all_parameter_scores.items():
            model_tensor = model_states[tensor_name]
            # Sanity check.
            assert model_tensor.shape == param_scores.shape

            self._check_unique_score(param_scores=param_scores)

            # Use score_threshold to produce sparse tensors.
            model_tensor_sparse = torch.where(param_scores >= score_threshold, input=model_tensor, other=0)
            # Tensors without any parameter or whose parameter values are all zero after thresholding
            # will not be exchanged, so we discard them.
            if not (model_tensor_sparse.shape == torch.Size([]) or (model_tensor_sparse == 0).all()):
                (
                    selected_parameters,
                    selected_indices,
                    tensor_shape,
                ) = self.parameter_packer.extract_coo_info_from_dense(model_tensor_sparse)
                selected_parameters_all_tensors.append(selected_parameters)
                selected_indices_all_tensors.append(selected_indices)
                tensor_shapes.append(tensor_shape)
                tensor_names.append(tensor_name)

        log(INFO, f"Sparsity level used to select parameters for exchange: {self.sparsity_level}")
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

        # Sanity check.
        assert len(selected_parameters) == len(indices) == len(shapes) == len(names) and len(names) > 0
        for param_values, param_indices, param_shape, param_name in zip(selected_parameters, indices, shapes, names):
            # Use parameter values, indices, and shape to create
            # a sparse coo tensor, which is then converted to a dense tensor
            # to allow for loading.
            param_coo = torch.sparse_coo_tensor(
                indices=torch.tensor(param_indices.T),
                values=torch.tensor(param_values),
                size=torch.Size(param_shape),
            )
            param_dense = param_coo.to_dense()
            current_state[param_name] = param_dense

        model.load_state_dict(current_state, strict=True)
