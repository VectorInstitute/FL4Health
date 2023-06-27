from logging import INFO
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.utils.metrics import AverageMeter, Meter, Metric


class BasicClient(NumpyFlClient):
    """
    This client implements a very basic flow where training is done by a specified number of steps and validation
    occurs over the full validation set. There are no special client side optimization changes, just the standard
    optimization flow.
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.model: nn.Module
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_examples: Dict[str, int]
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Set the model weights and initialize the correct weights with the parameter exchanger.
        super().set_parameters(parameters, config)
        # Saving the initial weights and detaching them so that we don't compute gradients with respect to the
        # tensors. These are used to form the FedProx loss.
        self.initial_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.model.parameters()
        ]

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        meter = AverageMeter(self.metrics, "train_meter")
        self.set_parameters(parameters, config)
        local_steps = self.narrow_config_type(config, "local_steps", int)
        # Currently uses training by epoch.
        metric_values = self.train_by_steps(local_steps, meter)
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            metric_values,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        meter = AverageMeter(self.metrics, "val_meter")
        loss, metric_values = self.validate(meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def train_by_steps(
        self,
        steps: int,
        meter: Meter,
    ) -> Dict[str, Scalar]:
        self.model.train()
        running_loss = 0.0
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
            preds = self.model(input)
            loss = self.criterion(preds, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            meter.update(preds, target)

        running_loss = running_loss / steps
        metrics = meter.compute()
        train_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            f"Client Training Losses: {running_loss} \n" f"Client Training Metrics: {train_metric_string}",
        )
        return metrics

    def validate(self, meter: Meter) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                pred = self.model(input)
                loss = self.criterion(pred, target)

                running_loss += loss.item()
                meter.update(pred, target)

        running_loss = running_loss / len(self.val_loader)
        metrics = meter.compute()
        val_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            "\n" f"Client Validation Losses: {running_loss} \n" f"Client validation Metrics: {val_metric_string}",
        )
        self._maybe_checkpoint(running_loss)
        return running_loss, metrics
