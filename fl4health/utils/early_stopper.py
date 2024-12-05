from logging import INFO
from pathlib import Path
from typing import Any, Callable, Optional, Type

import torch.nn as nn
from flwr.common.logger import log
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from fl4health.checkpointing.checkpointer import PerRoundCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.metrics import MetricManager
from fl4health.utils.snapshotter import (
    LRSchedulerSnapshotter,
    NumberSnapshotter,
    OptimizerSnapshotter,
    SerizableObjectSnapshotter,
    Snapshotter,
    T,
    TorchModuleSnapshotter,
)


class EarlyStopper:
    def __init__(
        self,
        client: BasicClient,
        patience: int = -1,
        interval_steps: int = 5,
        snapshot_dir: Optional[Path] = None,
    ) -> None:

        self.client = client

        self.patience = patience
        self.counte_down = patience
        self.interval_steps = interval_steps

        self.best_score: Optional[float] = None
        self.snapshot_ckpt: dict[str, Any] = {}

        self.default_snapshot_args: dict = {
            "model": (TorchModuleSnapshotter(self.client), nn.Module),  # for nn.Module we only need state_dict
            "optimizers": (OptimizerSnapshotter(self.client), Optimizer),  # dict of optimizers we only need state_dict
            "lr_schedulers": (
                LRSchedulerSnapshotter(self.client),
                _LRScheduler,
            ),  # dict of schedulers we only need state_dict
            "learning_rate": (NumberSnapshotter(self.client), float),  # number we can copy
            "total_steps": (NumberSnapshotter(self.client), int),  # number we can copy
            "total_epochs": (NumberSnapshotter(self.client), int),  # number we can copy
            "reports_manager": (
                SerizableObjectSnapshotter(self.client),
                ReportsManager,
            ),  # Class of objects we can copy
            "train_loss_meter": (
                SerizableObjectSnapshotter(self.client),
                TrainingLosses,
            ),  # Class of objects we can copy
            "train_metric_manager": (
                SerizableObjectSnapshotter(self.client),
                MetricManager,
            ),  # Class of objects we can copy
        }

        if snapshot_dir is not None:
            self.checkpointer = PerRoundCheckpointer(snapshot_dir, Path(f"temp_{self.client.client_name}.pt"))

    def add_default_snapshot_arg(
        self, name: str, snapshot_class: Callable[[BasicClient], Snapshotter], input_type: Type[T]
    ) -> None:
        self.default_snapshot_args.update({name: (snapshot_class(self.client), input_type)})

    def delete_default_snapshot_arg(self, name: str) -> None:
        del self.default_snapshot_args[name]

    def save_snapshot(self) -> None:
        """
        Saves checkpoint dict consisting of client name, total steps, lr schedulers,
            metrics reporter and optimizers state. Method can be overridden to augment saved checkpointed state.
        """
        for arg, (snapshotter_function, expected_type) in self.default_snapshot_args.items():
            self.snapshot_ckpt[arg] = snapshotter_function.save(arg, expected_type)

        if self.checkpointer is not None:
            self.checkpointer.save_checkpoint(self.snapshot_ckpt)
            self.snapshot_ckpt.clear()

        log(
            INFO,
            f"Saving client temp best state to checkpoint at {self.checkpointer.checkpoint_path}",
        )

    def load_snapshot(self) -> None:
        """
        Load checkpoint dict consisting of client name, total steps, lr schedulers, metrics
            reporter and optimizers state. Method can be overridden to augment loaded checkpointed state.
        """
        assert self.checkpointer.checkpoint_exists() or self.snapshot_ckpt != {}, "No checkpoint to load"

        if self.checkpointer.checkpoint_exists():
            self.snapshot_ckpt = self.checkpointer.load_checkpoint()

        for arg, (snapshotter_function, expected_type) in self.default_snapshot_args.items():
            snapshotter_function.load(self.snapshot_ckpt, arg, expected_type)

    def should_stop(self) -> bool:
        """
        Determine if the client should stop training based on early stopping criteria.

        Returns:
            bool: True if training should stop, otherwise False.
        """
        # Perform validation or testing and retrieve validation loss and metrics
        val_loss, _ = self.client._validate_or_test(
            loader=self.client.val_loader,
            loss_meter=self.client.val_loss_meter,
            metric_manager=self.client.val_metric_manager,
            logging_mode=LoggingMode.EARLY_STOP_VALIDATION,
            include_losses_in_metrics=False,  # TODO: Confirm this behavior
        )

        # If validation loss is not available, continue training
        if val_loss is None:
            return False

        # Update best score if it's the first evaluation
        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.count_down = self.patience  # Reset patience counter
            self.save_snapshot()
            return False

        # Reduce patience counter and check for early stopping
        self.count_down -= 1
        if self.count_down == 0:
            self.load_snapshot()
            return True

        return False
