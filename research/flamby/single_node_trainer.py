import os
from logging import INFO

import torch
from flwr.common.logger import log
from flwr.common.typing import Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.metrics.metric_managers import MetricManager


class SingleNodeTrainer:
    def __init__(
        self,
        device: torch.device,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        self.device = device
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        # This is called the "server model" so that it can be found by the evaluate_on_holdout.py script
        checkpoint_name = "server_best_model.pkl"
        self.checkpointer = BestLossTorchModuleCheckpointer(checkpoint_dir, checkpoint_name)
        self.dataset_dir = dataset_dir
        self.model: nn.Module
        self.criterion: _Loss
        self.optimizer: torch.optim.Optimizer
        self.train_loader: DataLoader
        self.val_loader: DataLoader

    def _maybe_checkpoint(self, loss: float, metrics: dict[str, Scalar]) -> None:
        if self.checkpointer:
            self.checkpointer.maybe_checkpoint(self.model, loss, metrics)

    def _handle_reporting(
        self,
        loss: float,
        metrics_dict: dict[str, Scalar],
        is_validation: bool = False,
    ) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Centralized {metric_prefix} Loss: {loss} \nCentralized {metric_prefix} Metrics: {metric_string}",
        )

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # forward pass on the model
        preds = self.model(input)
        loss = self.criterion(preds, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, {"predictions": preds}

    def train_by_epochs(
        self,
        epochs: int,
        train_metric_mngr: MetricManager,
        val_metric_mngr: MetricManager,
    ) -> None:
        self.model.train()

        for local_epoch in range(epochs):
            train_metric_mngr.clear()
            running_loss = 0.0
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                batch_loss, preds = self.train_step(input, target)
                running_loss += batch_loss.item()
                train_metric_mngr.update(preds, target)

            log(INFO, f"Local Epoch: {local_epoch}")
            running_loss = running_loss / len(self.train_loader)
            metrics = train_metric_mngr.compute()
            self._handle_reporting(running_loss, metrics)

            # After each epoch run a validation pass
            self.validate(val_metric_mngr)

    def validate(self, val_metric_mngr: MetricManager) -> None:
        self.model.eval()
        running_loss = 0.0
        val_metric_mngr.clear()

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                preds = {"predictions": self.model(input)}
                batch_loss = self.criterion(preds["predictions"], target)
                running_loss += batch_loss.item()
                val_metric_mngr.update(preds, target)

        running_loss = running_loss / len(self.val_loader)
        metrics = val_metric_mngr.compute()
        self._handle_reporting(running_loss, metrics, is_validation=True)
        self._maybe_checkpoint(running_loss, metrics)
