from logging import INFO
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.utils.metrics import AverageMeter, Metric


class ApflClient(NumpyFlClient):
    def __init__(
        self,
        data_path: Path,
        metrics: List[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, device)
        self.metrics = metrics
        self.model: APFLModule
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_examples: Dict[str, int]
        self.criterion: _Loss
        self.local_optimizer: torch.optim.Optimizer
        self.global_optimizer: torch.optim.Optimizer

    def is_start_of_local_training(self, epoch: int, step: int) -> bool:
        return epoch == 0 and step == 0

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        metric_values = self.train(local_epochs)
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

    def train(
        self,
        epochs: int,
    ) -> Dict[str, Scalar]:
        self.model.train()
        for epoch in range(epochs):
            loss_dict = {"personal": 0.0, "local": 0.0, "global": 0.0}

            global_meter = AverageMeter(self.metrics, "global")
            local_meter = AverageMeter(self.metrics, "local")
            personal_meter = AverageMeter(self.metrics, "personal")
            for step, (input, target) in enumerate(self.train_loader):
                # Mechanics of training loop follow from original implementation
                # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
                input, target = input.to(self.device), target.to(self.device)

                # Forward pass on global model and update global parameters
                self.global_optimizer.zero_grad()
                global_pred = self.model(input, personal=False)["global"]
                global_loss = self.criterion(global_pred, target)
                global_loss.backward()
                self.global_optimizer.step()

                # Make sure gradients are zero prior to foward passes of global and local model
                # to generate personalized predictions
                # NOTE: We zero the global optimizer grads because they are used (after the backward calculation below)
                # to update the scalar alpha (see update_alpha() where .grad is called.)
                self.global_optimizer.zero_grad()
                self.local_optimizer.zero_grad()

                # Personal predictions are generated as a convex combination of the output
                # of local and global models
                pred_dict = self.model(input, personal=True)
                personal_pred, local_pred = pred_dict["personal"], pred_dict["local"]

                # Parameters of local model are updated to minimize loss of personalized model
                personal_loss = self.criterion(personal_pred, target)
                personal_loss.backward()
                self.local_optimizer.step()

                with torch.no_grad():
                    local_loss = self.criterion(local_pred, target)

                # Only update alpha if it is the first epoch and first step of training
                # and adaptive alpha is true
                if self.is_start_of_local_training(epoch, step) and self.model.adaptive_alpha:
                    self.model.update_alpha()

                loss_dict["local"] += local_loss.item()
                loss_dict["global"] += global_loss.item()
                loss_dict["personal"] += personal_loss.item()

                global_meter.update(global_pred, target)
                local_meter.update(local_pred, target)
                personal_meter.update(personal_pred, target)

            loss_dict = {key: val / len(self.train_loader) for key, val in loss_dict.items()}

        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: Dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        train_loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        train_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            f"Epoch: {epoch} alpha: {self.model.alpha} \n"
            f"Client Training Losses: {train_loss_string} \n"
            f"Client Training Metrics: {train_metric_string}",
        )

        return metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        global_meter = AverageMeter(self.metrics, "global")
        local_meter = AverageMeter(self.metrics, "local")
        personal_meter = AverageMeter(self.metrics, "personal")
        loss_dict = {"global": 0.0, "personal": 0.0, "local": 0.0}

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                global_pred = self.model(input, personal=False)["global"]
                global_loss = self.criterion(global_pred, target)

                pred_dict = self.model(input, personal=True)
                personal_pred, local_pred = pred_dict["personal"], pred_dict["local"]
                personal_loss = self.criterion(personal_pred, target)
                local_loss = self.criterion(local_pred, target)

                loss_dict["global"] += global_loss.item()
                loss_dict["personal"] += personal_loss.item()
                loss_dict["local"] += local_loss.item()

                global_meter.update(global_pred, target)
                local_meter.update(local_pred, target)
                personal_meter.update(personal_pred, target)

        loss_dict = {key: val / len(self.val_loader) for key, val in loss_dict.items()}
        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: Dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        val_loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        val_metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics.items()])
        log(
            INFO,
            "\n" f"Client Validation Losses: {val_loss_string} \n" f"Client validation Metrics: {val_metric_string}",
        )

        return loss_dict["global"], metrics
