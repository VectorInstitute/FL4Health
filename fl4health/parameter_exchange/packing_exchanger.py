from typing import Generic, TypeVar

from flwr.common.typing import NDArrays

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPacker


T = TypeVar("T")


class FullParameterExchangerWithPacking(FullParameterExchanger, Generic[T]):
    def __init__(self, parameter_packer: ParameterPacker[T]) -> None:
        """
        Parameter exchanger for when sending the entire set of model weights between the client and server with
        potential side information packed in as well.

        Args:
            parameter_packer (ParameterPacker[T]): Parameter packer used to pack and unpack auxiliary information
                alongside the model weights.
        """
        super().__init__()
        self.parameter_packer = parameter_packer

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        return self.parameter_packer.pack_parameters(model_weights, additional_parameters)

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, T]:
        return self.parameter_packer.unpack_parameters(packed_parameters)
