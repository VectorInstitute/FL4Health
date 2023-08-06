from logging import INFO
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.utils.metrics import AverageMeter, Meter, Metric

FedProxTrainStepOutputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class FedProxClient(NumpyFlClient):
    """
    This client implements the FedProx algorithm from Federated Optimization in Heterogeneous Networks. The idea is
    fairly straightforward. The local loss for each client is augmented with a norm on the difference between the
    local client weights during training (w) and the initial globally shared weights (w^t).
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
        # This should be the "basic loss function" to be optimized. We'll add in the proximal term to this base
        # loss. That is, the FedProx loss becomes criterion + \mu \Vert w - w^t \Vert ^2
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer
        self.parameter_exchanger: ParameterExchangerWithPacking[Dict[str, NDArrays]]
        self.initial_tensors: List[torch.Tensor]
        self.proximal_weight: float = 0.0
        self.total_epochs = 0
        self.total_steps = 0

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
        """
        Assumes that the parameters being passed contain model parameters concatenated with
        proximal weight control variates. They are unpacked for the clients to use in training.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, server_extra_variables = self.parameter_exchanger.unpack_parameters(parameters)

        # Setting the proximal weight
        for key, value in server_extra_variables.items():
            setattr(self, key, value[0])
        self.server_model_state = server_model_state
        self.parameter_exchanger.pull_parameters(server_model_state, self.model, config)

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
        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        # Currently uses training by epoch.
        metric_values = self.train_by_epochs(current_server_round, local_epochs, meter)

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
        meter = AverageMeter(self.metrics, "val_meter")
        loss, metric_values = self.validate(current_server_round, meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def _handle_reporting(
        self,
        to_log: Dict[str, Any],
        meter: Meter,
        loss_dict: Dict[str, float],
        steps_taken: int,
        is_validation: bool = False,
    ) -> Tuple[Dict[str, Scalar], Dict[str, float]]:
        # Average loss per step per loss component
        loss_dict = {key: val / steps_taken for key, val in loss_dict.items()}

        metrics = meter.compute()
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Client {metric_prefix} Losses: {loss_string} \n" f"Client {metric_prefix} Metrics: {metric_string}",
        )
        to_log.update(loss_dict)
        to_log.update(metrics)
        self._maybe_log_metrics(to_log)
        return metrics, loss_dict

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> FedProxTrainStepOutputs:
        # forward pass on the model
        preds = self.model(input)
        vanilla_loss = self.criterion(preds, target)
        proximal_loss = self.get_proximal_loss()
        fed_prox_loss = vanilla_loss + proximal_loss

        self.optimizer.zero_grad()
        fed_prox_loss.backward()
        self.optimizer.step()

        return vanilla_loss, proximal_loss, fed_prox_loss, preds

    def train_by_steps(
        self,
        current_server_round: int,
        steps: int,
        meter: Meter,
    ) -> Dict[str, Scalar]:
        self.model.train()
        loss_dict = {"train_vanilla_loss": 0.0, "train_proximal_loss": 0.0, "train_total_loss": 0.0}
        meter.clear()
        train_iterator = iter(self.train_loader)

        for _ in range(steps):
            self.total_steps += 1
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            input, target = input.to(self.device), target.to(self.device)
            vanilla_loss, proximal_loss, fed_prox_loss, preds = self.train_step(input, target)

            loss_dict["train_vanilla_loss"] += vanilla_loss.item()
            loss_dict["train_proximal_loss"] += proximal_loss.item()
            loss_dict["train_total_loss"] += fed_prox_loss.item()

            meter.update(preds, target)

        custom_log: Dict[str, Any] = {"step": self.total_steps, "server_round": current_server_round}
        metrics, _ = self._handle_reporting(custom_log, meter, loss_dict, steps)

        # return final training metrics
        return metrics

    def train_by_epochs(
        self,
        current_server_round: int,
        epochs: int,
        meter: Meter,
    ) -> Dict[str, Scalar]:
        self.model.train()

        for local_epoch in range(epochs):
            meter.clear()
            self.total_epochs += 1
            loss_dict = {"train_vanilla_loss": 0.0, "train_proximal_loss": 0.0, "train_total_loss": 0.0}
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                vanilla_loss, proximal_loss, fed_prox_loss, preds = self.train_step(input, target)

                loss_dict["train_vanilla_loss"] += vanilla_loss.item()
                loss_dict["train_proximal_loss"] += proximal_loss.item()
                loss_dict["train_total_loss"] += fed_prox_loss.item()

                meter.update(preds, target)

            log(INFO, f"Local Epoch: {local_epoch}")
            custom_log: Dict[str, Any] = {"epoch": self.total_epochs, "server_round": current_server_round}
            metrics, _ = self._handle_reporting(custom_log, meter, loss_dict, len(self.train_loader))

        # Return final training metrics
        return metrics

    def validate(self, current_server_round: int, meter: Meter) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        loss_dict = {"val_vanilla_loss": 0.0, "val_proximal_loss": 0.0, "val_total_loss": 0.0}
        meter.clear()

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

        custom_log: Dict[str, Any] = {"server_round": current_server_round}
        metrics, loss_per_step = self._handle_reporting(
            custom_log, meter, loss_dict, len(self.val_loader), is_validation=True
        )

        val_loss_per_step = loss_per_step["val_total_loss"]
        self._maybe_checkpoint(val_loss_per_step)

        return val_loss_per_step, metrics
