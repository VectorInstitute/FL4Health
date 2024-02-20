from abc import abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import torch.nn as nn
from flwr.common.typing import NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPacker

T = TypeVar("T")


class PartialParameterExchanger(ParameterExchanger, Generic[T]):
    def __init__(self, parameter_packer: ParameterPacker[T]) -> None:
        super().__init__()
        self.parameter_packer = parameter_packer

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        return self.parameter_packer.pack_parameters(model_weights, additional_parameters)

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, T]:
        return self.parameter_packer.unpack_parameters(packed_parameters)

    @abstractmethod
    def select_parameters(
        self,
        model: nn.Module,
        initial_model: Optional[nn.Module] = None,
    ) -> Tuple[NDArrays, T]:
        raise NotImplementedError
