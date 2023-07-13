from logging import INFO
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.utils.metrics import AverageMeter, Meter, Metric

LocalLoss = torch.Tensor
GlobalLoss = torch.Tensor
PersonalLoss = torch.Tensor

LocalPreds = torch.Tensor
GlobalPreds = torch.Tensor
PersonalPreds = torch.Tensor

ApflTrainStepOutputs = Tuple[LocalLoss, GlobalLoss, PersonalLoss, LocalPreds, GlobalPreds, PersonalPreds]


class ApflClient(NumpyFlClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
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

        # Default APFL uses an average meter
        global_meter = AverageMeter(self.metrics, "global")
        local_meter = AverageMeter(self.metrics, "local")
        personal_meter = AverageMeter(self.metrics, "personal")
        # By default the APFL client trains by epochs
        metric_values = self.train_by_epoch(local_epochs, global_meter, local_meter, personal_meter)
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
        # Default APFL uses an average meter
        global_meter = AverageMeter(self.metrics, "global")
        local_meter = AverageMeter(self.metrics, "local")
        personal_meter = AverageMeter(self.metrics, "personal")
        loss, metric_values = self.validate(global_meter, local_meter, personal_meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def _handle_logging(
        self, loss_dict: Dict[str, float], metrics_dict: Dict[str, Scalar], is_validation: bool = False
    ) -> None:
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in loss_dict.items()])
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"alpha: {self.model.alpha} \n"
            f"Client {metric_prefix} Losses: {loss_string} \n"
            f"Client {metric_prefix} Metrics: {metric_string}",
        )

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> ApflTrainStepOutputs:
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

        return local_loss, global_loss, personal_loss, local_pred, global_pred, personal_pred

    def train_by_steps(
        self, steps: int, global_meter: Meter, local_meter: Meter, personal_meter: Meter
    ) -> Dict[str, Scalar]:
        self.model.train()
        loss_dict = {"personal": 0.0, "local": 0.0, "global": 0.0}
        global_meter.clear()
        local_meter.clear()
        personal_meter.clear()

        train_iterator = iter(self.train_loader)

        for step in range(steps):
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            # Mechanics of training loop follow from original implementation
            # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
            local_loss, global_loss, personal_loss, local_preds, global_preds, personal_preds = self.train_step(
                input, target
            )

            # Only update alpha if it is the first step of local training  and adaptive alpha is true
            if step == 0 and self.model.adaptive_alpha:
                self.model.update_alpha()

            loss_dict["local"] += local_loss.item()
            loss_dict["global"] += global_loss.item()
            loss_dict["personal"] += personal_loss.item()

            global_meter.update(global_preds, target)
            local_meter.update(local_preds, target)
            personal_meter.update(personal_preds, target)

        loss_dict = {key: val / steps for key, val in loss_dict.items()}
        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: Dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        log(INFO, f"Performed {steps} Steps of Local training")
        self._handle_logging(loss_dict, metrics)

        # return final training metrics
        return metrics

    def train_by_epoch(
        self, epochs: int, global_meter: Meter, local_meter: Meter, personal_meter: Meter
    ) -> Dict[str, Scalar]:
        self.model.train()
        for epoch in range(epochs):
            loss_dict = {"personal": 0.0, "local": 0.0, "global": 0.0}
            global_meter.clear()
            local_meter.clear()
            personal_meter.clear()

            for step, (input, target) in enumerate(self.train_loader):
                # Mechanics of training loop follow from original implementation
                # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
                local_loss, global_loss, personal_loss, local_preds, global_preds, personal_preds = self.train_step(
                    input, target
                )

                # Only update alpha if it is the first epoch and first step of training
                # and adaptive alpha is true
                if self.is_start_of_local_training(epoch, step) and self.model.adaptive_alpha:
                    self.model.update_alpha()

                loss_dict["local"] += local_loss.item()
                loss_dict["global"] += global_loss.item()
                loss_dict["personal"] += personal_loss.item()

                global_meter.update(global_preds, target)
                local_meter.update(local_preds, target)
                personal_meter.update(personal_preds, target)

            loss_dict = {key: val / len(self.train_loader) for key, val in loss_dict.items()}

        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: Dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        log(INFO, f"Performed {epochs} Epochs of Local training")
        self._handle_logging(loss_dict, metrics)

        return metrics

    def validate(
        self, global_meter: Meter, local_meter: Meter, personal_meter: Meter
    ) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        loss_dict = {"global": 0.0, "personal": 0.0, "local": 0.0}
        global_meter.clear()
        local_meter.clear()
        personal_meter.clear()

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
        self._handle_logging(loss_dict, metrics, is_validation=True)
        self._maybe_checkpoint(loss_dict["personal"])
        return loss_dict["personal"], metrics
