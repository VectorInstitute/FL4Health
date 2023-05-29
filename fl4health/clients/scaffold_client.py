from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithControlVariates
from fl4health.utils.metrics import AverageMeter, Metric


class ScaffoldClient(NumpyFlClient):
    """
    Federated Learning Client for Scaffold strategy.

    Implementation based on https://arxiv.org/pdf/1910.06378.pdf.
    """

    def __init__(self, data_path: Path, metrics: List[Metric], device: torch.device) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.client_control_variates: Optional[NDArrays] = None  # c_i in paper
        self.client_control_variates_updates: Optional[NDArrays] = None  # delta_c_i in paper
        self.server_control_variates: Optional[NDArrays] = None  # c in paper
        self.model: nn.Module
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.criterion: _Loss
        self.optimizer: torch.optim.SGD  # Scaffold require vanilla SGD as optimizer
        self.learning_rate_local: float  # eta_l in paper
        self.server_model_weights: Optional[NDArrays] = None  # x in paper
        self.num_examples: Dict[str, int]
        self.parameter_exchanger: ParameterExchangerWithControlVariates

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
        if not self.initialized:
            self.setup_client(config)

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
        Packs the parameters and control variartes into a single NDArrays to be sent to the server for aggregation
        """
        assert self.model is not None and self.parameter_exchanger is not None

        model_weights = self.parameter_exchanger.push_parameters(self.model, config)

        # Weights and control variates updates sent to server for aggregation
        # Control variates updates sent because only client has access to previous client control variate
        # Therefore it can only be computed locally
        assert self.client_control_variates_updates is not None
        packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.client_control_variates_updates)
        return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with
        server control variates. They are unpacked for the clients to use in training
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_weights, server_control_variates = self.parameter_exchanger.unpack_parameters(parameters)
        self.server_control_variates = server_control_variates
        self.server_model_weights = server_model_weights
        self.parameter_exchanger.pull_parameters(server_model_weights, self.model, config)

        # If client control variates do not exist, initialize with zeros as per paper
        if self.client_control_variates is None:
            self.client_control_variates = [np.zeros_like(weight) for weight in self.server_model_weights]

    def update_control_variates(self, local_steps: int) -> None:
        """
        Updates local control variates along with the corresponding updates
        according to the option 2 in Equation 4 in https://arxiv.org/pdf/1910.06378.pdf
        To be called after weights of local model have been updated.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None
        assert self.server_model_weights is not None
        assert self.learning_rate_local is not None
        assert self.train_loader.batch_size is not None

        # y_i
        client_model_weights = [val.numpy() for val in self.model.state_dict().values()]

        # (x - y_i)
        delta_model_weights = self.compute_parameters_delta(self.server_model_weights, client_model_weights)

        # (c_i - c)
        delta_control_variates = self.compute_parameters_delta(
            self.client_control_variates, self.server_control_variates
        )

        updated_client_control_variates = self.compute_updated_control_variates(
            local_steps, delta_model_weights, delta_control_variates
        )
        self.client_control_variates_updates = self.compute_parameters_delta(
            updated_client_control_variates, self.client_control_variates
        )

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
            assert param.grad is not None
            tensor_type = param.grad.dtype
            update = torch.from_numpy(server_cv).type(tensor_type) - torch.from_numpy(client_cv).type(tensor_type)
            param.grad += update

    def compute_parameters_delta(self, params_1: NDArrays, params_2: NDArrays) -> NDArrays:
        """
        Computes elementwise difference of two lists of NDarray
        where elements in params_2 are subtracted from elements in params_1
        """
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]

        return parameter_delta

    def compute_updated_control_variates(
        self, local_steps: int, delta_model_weights: NDArrays, delta_control_variates: NDArrays
    ) -> NDArrays:
        """
        Computes the updated local control variates according to option 2 in Equation 4 of paper
        """

        # coef = 1 / (K * eta_l)
        scaling_coeffient = 1 / (local_steps * self.learning_rate_local)

        # c_i^plus = c_i - c + 1/(K*lr) * (x - y_i)
        updated_client_control_variates = [
            delta_control_variate + scaling_coeffient * delta_model_weight
            for delta_control_variate, delta_model_weight in zip(delta_control_variates, delta_model_weights)
        ]
        return updated_client_control_variates

    def train(
        self,
        local_steps: int,
    ) -> Dict[str, Scalar]:
        self.model.train()

        running_loss = 0.0
        # Pass loader to iterator so we can step through  train loader
        loader = iter(self.train_loader)
        meter = AverageMeter(self.metrics, "global")
        for step in range(local_steps):
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
