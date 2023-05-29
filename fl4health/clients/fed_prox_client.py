from logging import INFO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.utils.metrics import AverageMeter, Metric


class FedProxClient(NumpyFlClient):
    """
    This client implements the FedProx algorithm from Federated Optimization in Heterogeneous Networks. The idea is
    fairly straightforward. The local loss for each client is augmented with a norm on the difference between the
    local client weights during training (w) and the initial globally shared weights (w^t).
    """

    def __init__(
        self,
        data_path: Path,
        metrics: List[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.model: nn.Module
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_examples: Dict[str, int]
        # This should be the "basic loss function" to be optimized. We'll add in the proximal term to this base
        # loss. That is, the FedProx loss becomes criterion + \mu \Vert w - w^t \Vert ^2
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer
        self.proximal_weight: float = 0.1
        self.initial_tensors: List[torch.Tensor]
        self.total_epochs = 0

    def get_proximal_loss(self) -> torch.Tensor:
        assert self.initial_tensors is not None
        # Using state dictionary to ensure the same ordering as exchange
        model_weights = [layer_weights for layer_weights in self.model.parameters()]
        assert len(self.initial_tensors) == len(model_weights)

        layer_inner_products: List[torch.Tensor] = [
            torch.pow(torch.linalg.norm(initial_layer_weights - iteration_layer_weights), 2.0)
            for initial_layer_weights, iteration_layer_weights in zip(self.initial_tensors, model_weights)
        ]

        # network l2 inner product tensor
        # NOTE: Scaling by 1/2 is for consistency with the original fedprox paper.
        return (self.proximal_weight / 2.0) * torch.stack(layer_inner_products).sum()

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

        self.set_parameters(parameters, config)
        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        metric_values = self.train(current_server_round, local_epochs)
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
        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        loss, metric_values = self.validate(current_server_round)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def _maybe_log_train_metrics(
        self, server_round: int, losses: Dict[str, float], metrics: Dict[str, Scalar]
    ) -> None:
        if self.wandb_reporter:
            to_log: Dict[str, Any] = {"epoch": self.total_epochs}
            to_log["server_round"] = server_round
            to_log.update(losses)
            to_log.update(metrics)
            self.wandb_reporter.report_metrics(to_log)

    def _maybe_log_eval_metrics(self, server_round: int, losses: Dict[str, float], metrics: Dict[str, Scalar]) -> None:
        if self.wandb_reporter:
            to_log: Dict[str, Any] = {"server_round": server_round}
            to_log.update(losses)
            to_log.update(metrics)
            self.wandb_reporter.report_metrics(to_log)

    def train(
        self,
        current_server_round: int,
        epochs: int,
    ) -> Dict[str, Scalar]:
        for epoch in range(epochs):
            self.total_epochs += 1
            meter = AverageMeter(self.metrics, "train_meter")
            loss_dict = {"train_vanilla_loss": 0.0, "train_proximal_loss": 0.0, "train_total_loss": 0.0}
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                # forward pass on the model
                preds = self.model(input)
                vanilla_loss = self.criterion(preds, target)
                proximal_loss = self.get_proximal_loss()
                fed_prox_loss = vanilla_loss + proximal_loss

                self.optimizer.zero_grad()
                fed_prox_loss.backward()
                self.optimizer.step()

                loss_dict["train_vanilla_loss"] += vanilla_loss.item()
                loss_dict["train_proximal_loss"] += proximal_loss.item()
                loss_dict["train_total_loss"] += fed_prox_loss.item()

                meter.update(preds, target)

            # Average loss per step per loss component
            loss_dict = {key: val / len(self.train_loader) for key, val in loss_dict.items()}

            training_metrics = meter.compute()
            metrics: Dict[str, Scalar] = {**training_metrics}
            train_loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
            train_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
            log(
                INFO,
                f"Epoch: {epoch}\n"
                f"Client Training Losses: {train_loss_string} \n"
                f"Client Training Metrics: {train_metric_string}",
            )
            self._maybe_log_train_metrics(current_server_round, loss_dict, metrics)

        return metrics

    def validate(self, current_server_round: int) -> Tuple[float, Dict[str, Scalar]]:
        meter = AverageMeter(self.metrics, "val_meter")
        loss_dict = {"val_vanilla_loss": 0.0, "val_proximal_loss": 0.0, "val_total_loss": 0.0}

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                preds = self.model(input)
                vanilla_loss = self.criterion(preds, target)
                proximal_loss = self.get_proximal_loss()
                fed_prox_loss = vanilla_loss + proximal_loss

                loss_dict["val_vanilla_loss"] += vanilla_loss.item()
                loss_dict["val_proximal_loss"] += proximal_loss.item()
                loss_dict["val_total_loss"] += fed_prox_loss.item()

                meter.update(preds, target)

        # Average loss per step per loss component
        loss_dict = {key: val / len(self.val_loader) for key, val in loss_dict.items()}
        validation_metrics = meter.compute()
        metrics: Dict[str, Scalar] = {**validation_metrics}
        val_loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        val_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            "\n" f"Client Validation Losses: {val_loss_string} \n" f"Client validation Metrics: {val_metric_string}",
        )
        self._maybe_log_eval_metrics(current_server_round, loss_dict, metrics)
        return loss_dict["val_total_loss"], metrics
