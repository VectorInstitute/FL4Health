from abc import abstractmethod
from typing import Tuple

from flwr.common.typing import NDArrays

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger


class ParameterExchangerWithPacking(FullParameterExchanger):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: NDArrays) -> NDArrays:
        return model_weights + additional_parameters

    @abstractmethod
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        raise NotImplementedError


class ParameterExchangerWithControlVariates(ParameterExchangerWithPacking):
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        assert len(packed_parameters) % 2 == 0
        split_size = len(packed_parameters) % 2
        return packed_parameters[:split_size], packed_parameters[split_size:]


class ParameterExchangerWithClippingBit(ParameterExchangerWithPacking):
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        split_size = len(packed_parameters) - 1
        return packed_parameters[:split_size], packed_parameters[split_size:]
