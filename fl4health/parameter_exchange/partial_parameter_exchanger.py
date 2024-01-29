from abc import abstractmethod
from typing import Optional, Tuple, TypeVar

import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking

T = TypeVar("T")


class PartialParameterExchanger(ParameterExchangerWithPacking[T]):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        return self.parameter_packer.pack_parameters(model_weights, additional_parameters)

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, T]:
        return self.parameter_packer.unpack_parameters(packed_parameters)

    @abstractmethod
    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_parameters(
        self,
        model: nn.Module,
        initial_model: Optional[nn.Module] = None,
    ) -> Tuple[NDArrays, T]:
        raise NotImplementedError
