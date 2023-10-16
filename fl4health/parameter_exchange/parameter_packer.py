from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy as np
from flwr.common.typing import List, NDArrays

T = TypeVar("T")


class ParameterPacker(ABC, Generic[T]):
    @abstractmethod
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, T]:
        raise NotImplementedError


class ParameterPackerWithControlVariates(ParameterPacker[NDArrays]):
    def __init__(self, size_of_model_params: int) -> None:
        # Note model params exchanged and control variates can be different sizes, for example, when layers are frozen
        # or the state dictionary contains things like Batch Normalization layers.
        self.size_of_model_params = size_of_model_params
        super().__init__()

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: NDArrays) -> NDArrays:
        return model_weights + additional_parameters

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        return packed_parameters[: self.size_of_model_params], packed_parameters[self.size_of_model_params :]


class ParameterPackerWithClippingBit(ParameterPacker[float]):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: float) -> NDArrays:
        return model_weights + [np.array(additional_parameters)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, float]:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        clipping_bound = float(packed_parameters[split_size:][0])
        return model_parameters, clipping_bound


class ParameterPackerFedProx(ParameterPacker[float]):
    def pack_parameters(self, model_weights: NDArrays, extra_fedprox_variable: float) -> NDArrays:
        return model_weights + [np.array(extra_fedprox_variable)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, float]:
        # The last entry is extra packed fedprox variable
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        # The packed contents should have length 1
        packed_contents = packed_parameters[split_size:]
        assert len(packed_contents) == 1
        extra_fedprox_variable = float(packed_contents[0])
        return model_parameters, extra_fedprox_variable


class ParameterPackerWithLayerNames(ParameterPacker[List[str]]):
    def pack_parameters(self, model_weights: NDArrays, weights_names: List[str]) -> NDArrays:
        return model_weights + [np.array(weights_names)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, List[str]]:
        """
        Assumption: packed_parameters is a list containing model parameters followed by an NDArray that contains the
        corresponding names of those parameters.
        """
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        param_names = packed_parameters[split_size:][0].tolist()
        return model_parameters, param_names
