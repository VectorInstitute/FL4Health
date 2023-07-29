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
    
class ParameterPackerWithExtraVariables(ParameterPacker[List[float]]):
    def __init__(self, size_of_model_params: int) -> None:
        # Note model params exchanged and control variates can be different sizes, for example, when layers are frozen
        # or the state dictionary contains things like Batch Normalization layers.
        self.size_of_model_params = size_of_model_params
        super().__init__()

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: List[float]) -> NDArrays:
        return model_weights + additional_parameters

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, float]:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        model_parameters = packed_parameters[:self.size_of_model_params]
        variables = packed_parameters[self.size_of_model_params:]
        return model_parameters, variables


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
