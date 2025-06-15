from abc import ABC, abstractmethod
from typing import TypeVar

from flwr.common.typing import Config, NDArrays
from torch import nn


class ParameterExchanger(ABC):
    @abstractmethod
    def push_parameters(
        self, model: nn.Module, initial_model: nn.Module | None = None, config: Config | None = None
    ) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Config | None = None) -> None:
        raise NotImplementedError


ExchangerType = TypeVar("ExchangerType", bound=ParameterExchanger)
