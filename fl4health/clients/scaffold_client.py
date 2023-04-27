from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.utils.metrics import AverageMeter, Metric


class ScaffoldClient(NumpyFlClient):
    def __init__(self, data_path: Path, metrics: List[Metric], device: torch.device) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.client_control_variates: Union[NDArrays, None] = None  # c_i
        self.client_control_variates_updates = Union[NDArrays, None]  # c_update
        self.server_control_variates: Union[NDArrays, None] = None  # c
        self.model: nn.Module
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer
        self.learning_rate_control_variates: float
        self.server_model_weights: NDArrays  # x
        self.num_examples: Dict[str, int]

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        local_steps = self.narrow_config_type(config, "local_steps", int)
        metric_values = self.train(local_steps)

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            metric_values,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters, config)
        loss, metric_values = self.validate()
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Packs the parameters and control variartes into a single NDArray
        """
        model_weights = self.parameter_exchanger.push_parameters(self.model, config)
        return self.pack_parameters(model_weights)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with
        control variates.
        """
        server_model_weights, server_control_variates = self.unpack_parameters(parameters)
        self.server_control_variates = server_control_variates
        self.server_model_weights = server_model_weights
        self.parameter_exchanger.pull_parameters(server_model_weights, self.model, config)

        if self.client_control_variates is None:
            self.client_control_variates = [np.zeros_like(weight) for weight in self.server_model_weights]

    def pack_parameters(self, model_weights: NDArrays) -> NDArrays:
        """
        Extends the parameters list to include both the model weights and updates to the
        local control variate.
        """
        assert self.client_control_variates_updates is not None
        assert isinstance(model_weights, List) and isinstance(self.client_control_variates_updates, List)
        return model_weights + self.client_control_variates_updates

    def unpack_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        """
        Splits the passed parameter list into the model weights and server control variates.
        """
        split_size = len(parameters) // 2
        weights, control_variates = parameters[:split_size], parameters[split_size:]
        return weights, control_variates

    def update_control_variates(self, local_steps: int) -> None:
        """
        Updates local control variates along with the corresponding updates
        according to the option 2 in Equation 4 in https://arxiv.org/pdf/1910.06378.pdf
        To be called after weights of local model have been updated.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None
        assert self.server_model_weights is not None
        assert self.learning_rate_control_variates is not None
        assert self.train_loader.batch_size is not None

        # y_i
        client_model_weights = [val.numpy() for val in self.model.state_dict().values()]

        # (x - y_i)
        delta_model_weights = [
            server_model_weight - client_model_weight
            for client_model_weight, server_model_weight in zip(client_model_weights, self.server_model_weights)
        ]

        # coef = 1 / (K * lr)
        scaling_coeffient = 1 / (local_steps * self.learning_rate_control_variates)

        # c_i^plus = c_i - c + 1/(K*lr) * (x - y_i)
        updated_client_control_variates = [
            client_control_variate - server_control_variate + scaling_coeffient * delta_model_weight
            for client_control_variate, delta_model_weight, server_control_variate in zip(
                self.client_control_variates, delta_model_weights, self.server_control_variates
            )
        ]

        # c_delta = c_i^plus - c_i
        self.client_control_variates_updates = [
            updated_control_variate - current_control_variate
            for updated_control_variate, current_control_variate in zip(
                updated_client_control_variates, self.client_control_variates
            )
        ]

        # c_i = c_i^plus
        self.client_control_variates = updated_client_control_variates

    def modify_grad(self) -> None:
        """
        Modifies the gradient of the local model to correct for client drift.
        To be called after the gradients have been computed on a batch of data.
        Updates not applied to params until step is called on optimizer.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None

        for param, client_cv, server_cv in zip(
            self.model.parameters(), self.client_control_variates, self.server_control_variates
        ):
            # g_i(y_i) = g_i(y_i) - c_i + c
            param.grad.data += torch.from_numpy(server_cv).type(torch.float32) - torch.from_numpy(client_cv).type(
                torch.float32
            )

    def train(
        self,
        local_steps: int,
    ) -> Dict[str, Scalar]:
        self.model.train()

        running_loss = 0.0
        # Pass loader to iterator so we can step through  train loader
        loader = iter(self.train_loader)
        meter = AverageMeter(self.metrics, "global")
        for step in range(local_steps - 1):
            input, target = next(loader)
            input, target = input.to(self.device), target.to(self.device)

            # Forward pass on global model and update global parameters
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = self.criterion(pred, target)
            loss.backward()

            # modify grad to correct for client drift
            self.modify_grad()
            self.optimizer.step()

            running_loss += loss.item()
            meter.update(pred, target)

        running_loss = running_loss / len(self.train_loader)

        metrics = meter.compute()
        train_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            f"Client Training Losses: {loss} \n" f"Client Training Metrics: {train_metric_string}",
        )

        self.update_control_variates(local_steps)
        return metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        meter = AverageMeter(self.metrics, "global")
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

        return running_loss, metrics
