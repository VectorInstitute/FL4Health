from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.utils.typing import TorchFeatureType, TorchPredType, TorchTargetType


@runtime_checkable
class NumPyClientMinimalProtocol(Protocol):
    """A minimal protocol for NumPyClient with just essential methods."""

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        pass

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        pass

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        pass

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        pass

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        pass


@runtime_checkable
class BasicClientProtocolPreSetup(NumPyClientMinimalProtocol, Protocol):
    """A minimal protocol for BasicClient focused on methods."""

    device: torch.device
    initialized: bool

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

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        pass


@runtime_checkable
class BasicClientProtocol(BasicClientProtocolPreSetup, Protocol):
    """A minimal protocol for BasicClient focused on methods."""

    model: nn.Module
