import os
from logging import INFO
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from flwr.common.logger import log
from flwr.common.typing import Scalar
from monai.data.dataloader import DataLoader

from fl4health.checkpointing.checkpointer import PerEpochCheckpointer
from fl4health.utils.metrics import MetricManager


class SingleNodeTrainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_stub: str,
        run_name: str,
    ) -> None:
        self.device = device
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        # This is called the "server model" so that it can be found by the evaluate_on_holdout.py script
        checkpoint_name = "ckpt.pkl"
        self.checkpointer = PerEpochCheckpointer(checkpoint_dir, checkpoint_name)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.epoch: int
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        if not os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
            self.checkpointer.save_checkpoint(self.model, self.optimizer, 0)

    def _handle_reporting(
        self,
        loss: float,
        metrics_dict: Dict[str, Scalar],
        is_validation: bool = False,
    ) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        metric_prefix = "Validation" if is_validation else "Training"
        log(
            INFO,
            f"Centralized {metric_prefix} Loss: {loss} \n" f"Centralized {metric_prefix} Metrics: {metric_string}",
        )

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.model.train()
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
        val_metric_mngr: MetricManager
    ) -> None:

        while True:
            self.model, self.optimizer, self.epoch = self.checkpointer.load_checkpoint()
            if epochs <= self.epoch:
                break

            train_metric_mngr.clear()
            val_metric_mngr.clear()

            running_loss = 0.0
            for input, target in self.train_loader:
                input, target = input.as_tensor().to(self.device), target.as_tensor().to(self.device)
                batch_loss, preds = self.train_step(input, target)
                running_loss += batch_loss.item()
                train_metric_mngr.update(preds, target)

            log(INFO, f"Local Epoch: {str(self.epoch)}")
            running_loss = running_loss / len(self.train_loader)
            metrics = train_metric_mngr.compute()
            self._handle_reporting(running_loss, metrics)

            # After each epoch run a validation pass
            self.validate(val_metric_mngr)

            self.checkpointer.save_checkpoint(self.model, self.optimizer, self.epoch + 1)

    def validate(self, val_metric_mngr: MetricManager) -> None:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.as_tensor().to(self.device), target.as_tensor().to(self.device)

                preds = self.model(input)
                batch_loss = self.criterion(preds, target)
                running_loss += batch_loss.item()
                val_metric_mngr.update({"predictions": preds}, target)

        running_loss = running_loss / len(self.val_loader)
        metrics = val_metric_mngr.compute()
        self._handle_reporting(running_loss, metrics, is_validation=True)