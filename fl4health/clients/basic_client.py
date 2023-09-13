from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wanb import ClientWandBReporter
from fl4health.utils.metrics import AccumulationMeter, AverageMeter, Meter, MeterType, Metric


class BasicClient(NumpyFlClient):
    """
    Base FL Client with functionality to train, evaluate, log, report and checkpoint.
    User is only
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        meter_type: MeterType = MeterType.AVERAGE,
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.current_losses: Optional[Dict[str, float]] = None
        self.current_meter: Optional[Meter] = None
        self.use_wandb_reporter = use_wandb_reporter
        self.checkpointer = checkpointer
        self.meter_type = meter_type

        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_train_samples: int
        self.num_val_samples: int

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Set the model weights and initialize the correct weights with the parameter exchanger.
        super().set_parameters(parameters, config)

    def process_config(self, config: Config) -> Tuple[Union[int, None], Union[int, None], int]:
        """
        Method to ensure the required keys are present in config and extracts the values.
        """
        if "current_server_round" not in config:
            raise ValueError("Config must contain current_server_round key")
        current_server_round = self.narrow_config_type(config, "current_server_round", int)

        if ("local_epochs" in config) and ("local_steps" in config):
            raise ValueError("Config must contain one of local_epochs or local_steps key but not both")
        elif "local_epochs" in config:
            local_epochs = self.narrow_config_type(config, "local_epochs", int)
            local_steps = None
        elif "local_steps" in config:
            local_steps = self.narrow_config_type(config, "local_steps", int)
            local_epochs = None
        else:
            raise ValueError("Config must contain one of local_epochs or local_steps key but not both")

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)

        if local_epochs is not None:
            losses, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = self.num_train_samples * local_epochs  # total steps over training round
        else:
            assert isinstance(local_steps, int)
            losses, metrics = self.train_by_steps(local_steps, current_server_round)

        losses_float: Dict[str, float] = {key: float(val) for key, val in losses.items()}

        # Store current losses (Used by FedProx Client)
        self.current_losses = losses_float

        # Update model after train round (Used by Scaffold and DP-Scaffold Client)
        self.update_after_train(local_steps)

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        loss, metric_values = self.validate()
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_val_samples,
            metric_values,
        )

    def _handle_logging(
        self, loss_dict: Dict[str, Scalar], metrics_dict: Dict[str, Scalar], is_validation: bool = False
    ) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Client {metric_prefix} Losses: {loss_string} \n" f"Client {metric_prefix} Metrics: {metric_string}",
        )

    def update_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """
        Update current_losses attribute with new losses in loss_dict.
        If current_losses is None, initialize with keys of loss_dict and values 0
        """
        if self.current_losses is None:
            self.current_losses = {key: 0.0 for key in loss_dict.keys()}

        float_loss_dict = {key: val.item() for key, val in loss_dict.items()}
        self.current_losses = {key: val + float_loss_dict[key] for key, val in self.current_losses.items()}

    def update_meter(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update meter with predictions and targets.
        If no meter exists, meter is initialized based on the clients meter_type prior to updating.
        """
        if self.current_meter is None:
            if self.meter_type == MeterType.AVERAGE:
                self.current_meter = AverageMeter(self.metrics)
            else:
                self.current_meter = AccumulationMeter(self.metrics)

        self.current_meter.update(preds, target)

    def compute_metrics(self) -> Dict[str, Scalar]:
        assert self.current_meter is not None
        metrics = self.current_meter.compute()

        # Clear meter after computing metrics so we start fresh next time
        self.current_meter.clear()
        return metrics

    def compute_losses(self, step: int) -> Dict[str, Scalar]:
        assert self.current_losses is not None
        losses: Dict[str, Scalar] = {key: val / step for key, val in self.current_losses.items()}

        # Clear losses after computing metric so we start fresh next time
        self.current_losses = None
        return losses

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        Given input and target, generate predictions, compute loss, optionally update metrics if they exist.
        """
        # Clear gradients from optimizer if they exist
        self.optimizer.zero_grad()

        # Call user defined methods to get predictions and compute loss
        preds = self.predict(input)
        loss, loss_dict = self.compute_loss(preds, target)

        # Update losses and metrics
        loss_dict.update({"loss": loss})
        self.update_losses(loss_dict)
        self.update_meter(preds, target)

        # Compute backward pass and update paramters with optimizer
        loss.backward()
        self.optimizer.step()

    def val_step(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        Given input and target, compute loss, update loss and metrics
        """

        # Get preds and compute loss
        with torch.no_grad():
            preds = self.predict(input)
            loss, loss_dict = self.compute_loss(preds, target)

        # Updates losses and metrics
        loss_dict.update({"loss": loss})
        self.update_losses(loss_dict)
        self.update_meter(preds, target)

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, Scalar], Dict[str, Scalar]]:
        self.model.train()

        for local_epoch in range(epochs):
            for _, (input, target) in enumerate(self.train_loader):
                input, target = input.to(self.device), target.to(self.device)
                self.train_step(input, target)

            metrics = self.compute_metrics()
            losses = self.compute_losses(len(self.train_loader))

            log_string = (
                f"Local Epoch: {local_epoch} \t Server Round: {current_round}"
                if current_round is not None
                else f"Local Epoch: {local_epoch}"
            )

            log(INFO, log_string)
            self._handle_logging(losses, metrics)

        # Return final training metrics
        return losses, metrics

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, Scalar], Dict[str, Scalar]]:
        self.model.train()
        train_iterator = iter(self.train_loader)

        for _ in range(steps):
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            input, target = input.to(self.device), target.to(self.device)
            self.train_step(input, target)

        losses = self.compute_losses(steps)
        metrics = self.compute_metrics()

        if current_round is not None:
            log(INFO, f"Server Round: {current_round}")
        self._handle_logging(losses, metrics)

        return losses, metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                self.val_step(input, target)

        # Compute losses and metrics over validation set
        losses = self.compute_losses(len(self.val_loader))
        metrics = self.compute_metrics()
        self._handle_logging(losses, metrics, is_validation=True)
        assert isinstance(losses["loss"], float)

        # Checkpoint based on loss which is output of user defined compute_loss method
        self._maybe_checkpoint(losses["loss"])
        return losses["loss"], metrics

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties (train and validation dataset sample counts) of client.
        """
        if not self.initialized:
            self.setup_client(config)

        return {"num_train_samples": self.num_train_samples, "num_val_samples": self.num_val_samples}

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        """
        self.model = self.get_model(config)
        train_loader, val_loader = self.get_data_loaders(config)
        self.train_loader = train_loader
        self.val_loader = val_loader

        num_train_samples = len(self.train_loader.dataset)  # type: ignore
        num_val_samples = len(self.val_loader.dataset)  # type: ignore
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

        self.optimizer = self.get_optimizer(config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        if self.use_wandb_reporter:
            self.wandb_reporter = ClientWandBReporter.from_config(self.client_name, config)

        super().setup_client(config)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Returns Full Parameter Exchangers. Subclasses that require custom Parameter Exchangers can override this.
        """
        return FullParameterExchanger()

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader
        """
        raise NotImplementedError

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Method that the user defines returning loss and optionally a dictionairy with
        """
        raise NotImplementedError

    def get_optimizer(self, config: Config) -> Optimizer:
        """
        Method to be defined by user that returns the PyTorch optimizer used to train models locally
        """
        raise NotImplementedError

    def get_model(self, config: Config) -> nn.Module:
        """
        User defined method that Returns PyTorch model
        """
        raise NotImplementedError

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """
        User defined method to get predictions given input
        """
        raise NotImplementedError

    def update_after_train(self, local_steps: int) -> None:
        pass
