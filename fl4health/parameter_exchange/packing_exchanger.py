from typing import Generic, Tuple, TypeVar

from flwr.common.typing import NDArrays

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPacker

T = TypeVar("T")


class ParameterExchangerWithPacking(FullParameterExchanger, Generic[T]):
    def __init__(self, parameter_packer: ParameterPacker[T]) -> None:
        super().__init__()
        self.parameter_packer = parameter_packer

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        return self.parameter_packer.pack_parameters(model_weights, additional_parameters)

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, T]:
        return self.parameter_packer.unpack_parameters(packed_parameters)
