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
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: NDArrays) -> NDArrays:
        assert not isinstance(additional_parameters, float)
        return model_weights + additional_parameters

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        # Ensure that the packed parameters is even as a sanity check. Model paramers and control variates have same
        # size.
        assert len(packed_parameters) % 2 == 0
        split_size = len(packed_parameters) // 2
        return packed_parameters[:split_size], packed_parameters[split_size:]


class ParameterPackerWithClippingBit(ParameterPacker[float]):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: float) -> NDArrays:
        return model_weights + [np.array(additional_parameters)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, float]:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        clipping_bound = float(packed_parameters[split_size:][0])
        return model_parameters, clipping_bound


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
