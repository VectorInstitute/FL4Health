from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.utils.losses import EvaluationLosses, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


@runtime_checkable
class NumPyClientMinimalProtocol(Protocol):
    """A minimal protocol for NumPyClient with just essential methods."""

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        pass  # pragma: no cover

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        pass  # pragma: no cover

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        pass  # pragma: no cover

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        pass  # pragma: no cover

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        pass  # pragma: no cover


@runtime_checkable
class BasicClientProtocolPreSetup(NumPyClientMinimalProtocol, Protocol):
    """A minimal protocol for BasicClient focused on methods."""

    device: torch.device
    initialized: bool

    # Include only methods, not attributes that get initialized later
    def setup_client(self, config: Config) -> None:
        pass  # pragma: no cover

    def get_model(self, config: Config) -> nn.Module:
        pass  # pragma: no cover

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, ...]:
        pass  # pragma: no cover

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        pass  # pragma: no cover

    def get_criterion(self, config: Config) -> _Loss:
        pass  # pragma: no cover

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        pass  # pragma: no cover


@runtime_checkable
class BasicClientProtocol(BasicClientProtocolPreSetup, Protocol):
    """A minimal protocol for BasicClient focused on methods."""

    model: nn.Module
    optimizers: dict[str, torch.optim.Optimizer]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    criterion: _Loss

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        pass  # pragma: no cover

    def update_before_train(self, current_server_round: int) -> None:
        pass  # pragma: no cover

    def predict(self, input: TorchInputType) -> tuple[TorchPredType, TorchFeatureType]:
        pass  # pragma: no cover

    def transform_target(self, target: TorchTargetType) -> TorchTargetType:
        pass  # pragma: no cover

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        pass  # pragma: no cover

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        pass  # pragma: no cover

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        pass  # pragma: no cover
