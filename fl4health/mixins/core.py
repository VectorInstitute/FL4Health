from typing import Protocol, runtime_checkable

import torch.nn as nn
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@runtime_checkable
class NumPyClientMinimalProtocol(Protocol):
    """A minimal protocol for NumPyClient with just essential methods."""

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        pass

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        pass

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        pass


@runtime_checkable
class BasicClientProtocol(NumPyClientMinimalProtocol, Protocol):
    """A minimal protocol for BasicClient focused on methods."""

    # Include only methods, not attributes that get initialized later
    def setup_client(self, config: Config) -> None:
        pass

    def get_model(self, config: Config) -> nn.Module:
        pass

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, ...]:
        pass

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        pass

    def get_criterion(self, config: Config) -> _Loss:
        pass
