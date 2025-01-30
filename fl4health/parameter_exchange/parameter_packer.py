from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import torch
from flwr.common.typing import NDArray, NDArrays
from torch import Tensor

T = TypeVar("T")


class ParameterPacker(ABC, Generic[T]):
    @abstractmethod
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, T]:
        raise NotImplementedError


class ParameterPackerWithControlVariates(ParameterPacker[NDArrays]):
    def __init__(self, size_of_model_params: int) -> None:
        # Note model params exchanged and control variates can be different sizes, for example, when layers are frozen
        # or the state dictionary contains things like Batch Normalization layers.
        self.size_of_model_params = size_of_model_params
        super().__init__()

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: NDArrays) -> NDArrays:
        return model_weights + additional_parameters

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, NDArrays]:
        return packed_parameters[: self.size_of_model_params], packed_parameters[self.size_of_model_params :]


class ParameterPackerWithClippingBit(ParameterPacker[float]):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: float) -> NDArrays:
        return model_weights + [np.array(additional_parameters)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, float]:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        clipping_bound = packed_parameters[split_size:][0]
        return model_parameters, clipping_bound.item()


class ParameterPackerAdaptiveConstraint(ParameterPacker[float]):
    def pack_parameters(self, model_weights: NDArrays, extra_adaptive_variable: float) -> NDArrays:
        return model_weights + [np.array(extra_adaptive_variable)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, float]:
        # The last entry is an extra packed adaptive constraint variable (information to allow for adaptation)
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        # The packed contents should have length 1
        packed_contents = packed_parameters[split_size:]
        assert len(packed_contents) == 1
        extra_adaptive_variable = float(packed_contents[0])
        return model_parameters, extra_adaptive_variable


class ParameterPackerWithLayerNames(ParameterPacker[list[str]]):
    def pack_parameters(self, model_weights: NDArrays, weights_names: list[str]) -> NDArrays:
        return model_weights + [np.array(weights_names)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, list[str]]:
        """
        Assumption: packed_parameters is a list containing model parameters followed by an NDArray that contains the
        corresponding names of those parameters.
        """
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        param_names = packed_parameters[split_size:][0].tolist()
        return model_parameters, param_names


class SparseCooParameterPacker(ParameterPacker[tuple[NDArrays, NDArrays, list[str]]]):
    """
    This parameter packer is responsible for selecting an arbitrary set of parameters
    and then representing them in the sparse COO tensor format, which requires knowing
    the indices of the parameters within the tensor to which they belong,
    the shape of that tensor, and also the name of it.

    For more information on the sparse COO format and sparse tensors in PyTorch, please see the following
    two pages:
        1. https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
        2. https://pytorch.org/docs/stable/sparse.html

    """

    def pack_parameters(
        self, model_parameters: NDArrays, additional_parameters: tuple[NDArrays, NDArrays, list[str]]
    ) -> NDArrays:
        parameter_indices, tensor_shapes, tensor_names = additional_parameters
        return model_parameters + parameter_indices + tensor_shapes + [np.array(tensor_names)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, tuple[NDArrays, NDArrays, list[str]]]:
        # The names of the tensors is wrapped in a list, which is then transformed into an NDArrays of length 1
        # before packing.
        assert len(packed_parameters) % 3 == 1
        split_size = (len(packed_parameters) - 1) // 3
        model_parameters = packed_parameters[:split_size]
        parameter_indices = packed_parameters[split_size : (2 * split_size)]
        tensor_shapes = packed_parameters[(2 * split_size) : (3 * split_size)]
        tensor_names = packed_parameters[(3 * split_size) :][0].tolist()
        return model_parameters, (parameter_indices, tensor_shapes, tensor_names)

    @staticmethod
    def extract_coo_info_from_dense(x: Tensor) -> tuple[NDArray, NDArray, NDArray]:
        """
        Take a dense tensor x and extract the information required
        (namely, its nonzero values, their indices within the tensor, and the shape of x)
        in order to represent it in the sparse coo format.

        The results are converted to numpy arrays.

        Args:
            x (Tensor): Input dense tensor.

        Returns:
            tuple[NDArray, NDArray, NDArray]: The nonzero values of x,
            the indices of those values within x, and the shape of x.
        """
        selected_parameters = x[torch.nonzero(x, as_tuple=True)].cpu().numpy()
        selected_indices = torch.nonzero(x, as_tuple=False).cpu().numpy()
        tensor_shape = np.array(list(x.shape))
        return selected_parameters, selected_indices, tensor_shape
