from typing import Protocol, runtime_checkable

import torch
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.utils.losses import EvaluationLosses, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


@runtime_checkable
class NumPyClientMinimalProtocol(Protocol):
    """A minimal protocol for ``NumPyClient`` with just essential methods."""

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays: ...

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]: ...

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]: ...

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None: ...

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None: ...


@runtime_checkable
class FlexibleClientProtocolPreSetup(NumPyClientMinimalProtocol, Protocol):
    """A minimal protocol for ``BasicClient`` focused on methods."""

    device: torch.device
    initialized: bool

    # Include only methods, not attributes that get initialized later
    def setup_client(self, config: Config) -> None: ...

    def get_model(self, config: Config) -> nn.Module: ...

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, ...]: ...

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]: ...

    def get_criterion(self, config: Config) -> _Loss: ...

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: TorchFeatureType, target: TorchTargetType
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]: ...


@runtime_checkable
class FlexibleClientProtocol(FlexibleClientProtocolPreSetup, Protocol):
    """A minimal protocol for ``BasicClient`` focused on methods."""

    model: nn.Module
    optimizers: dict[str, torch.optim.Optimizer]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    criterion: _Loss

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None: ...

    def update_before_train(self, current_server_round: int) -> None: ...

    def _compute_preds_and_losses(
        self, model: nn.Module, optimizer: Optimizer, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]: ...

    def _apply_backwards_on_losses_and_take_step(
        self, model: nn.Module, optimizer: Optimizer, losses: TrainingLosses
    ) -> TrainingLosses: ...

    def _train_step_with_model_and_optimizer(
        self, model: nn.Module, optimizer: Optimizer, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]: ...

    def _val_step_with_model(
        self, model: nn.Module, input: TorchInputType, target: TorchTargetType
    ) -> tuple[EvaluationLosses, TorchPredType]: ...

    def predict_with_model(
        self, model: nn.Module, input: TorchInputType
    ) -> tuple[TorchPredType, TorchFeatureType]: ...

    def transform_target(self, target: TorchTargetType) -> TorchTargetType: ...

    def _transform_gradients_with_model(self, model: torch.nn.Module, losses: TrainingLosses) -> None: ...

    def transform_gradients(self, losses: TrainingLosses) -> None: ...

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses: ...

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]: ...

    def compute_evaluation_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses: ...
