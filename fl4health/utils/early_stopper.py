from __future__ import annotations

from collections.abc import Callable
from logging import INFO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from flwr.common.logger import log
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.logging import LoggingMode
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.metrics import MetricManager
from fl4health.utils.snapshotter import (
    AbstractSnapshotter,
    LRSchedulerSnapshotter,
    NumberSnapshotter,
    OptimizerSnapshotter,
    SerializableObjectSnapshotter,
    T,
    TorchModuleSnapshotter,
)

if TYPE_CHECKING:
    from fl4health.clients.basic_client import BasicClient


class EarlyStopper:
    def __init__(
        self,
        client: BasicClient,
        patience: int | None = 1,
        interval_steps: int = 5,
        snapshot_dir: Path | None = None,
    ) -> None:
        """
        Early stopping class is a plugin for the client that allows to stop local training based on the validation
        loss. At each training step this class saves the best state of the client and restores it if the client is
        stopped. If the client starts to overfit, the early stopper will stop the training process and restore the best
        state of the client before sending the model to the server.

        Args:
            client (BasicClient): The client to be monitored.
            patience (int, optional): Number of validation cycles to wait before stopping the training. If it is equal
                to None client never stops, but still loads the best state before sending the model to the server.
                Defaults to 1.
            interval_steps (int, optional): Specifies the frequency, in terms of training intervals, at which the early
                stopping mechanism should evaluate the validation loss. Defaults to 5.
            snapshot_dir (Path | None, optional): Rather than keeping best state in the memory we can checkpoint it to
                the given directory. If it is not given, the best state is kept in the memory. Defaults to None.
        """

        self.client = client

        self.patience = patience
        self.count_down = patience
        self.interval_steps = interval_steps

        self.best_score: float | None = None
        self.snapshot_ckpt: dict[str, tuple[AbstractSnapshotter, Any]] = {}

        self.snapshot_attrs: dict = {
            "model": (TorchModuleSnapshotter(self.client), nn.Module),
            "optimizers": (OptimizerSnapshotter(self.client), Optimizer),
            "lr_schedulers": (
                LRSchedulerSnapshotter(self.client),
                LRScheduler,
            ),
            "learning_rate": (NumberSnapshotter(self.client), float),
            "total_steps": (NumberSnapshotter(self.client), int),
            "total_epochs": (NumberSnapshotter(self.client), int),
            "reports_manager": (
                SerializableObjectSnapshotter(self.client),
                ReportsManager,
            ),
            "train_loss_meter": (
                SerializableObjectSnapshotter(self.client),
                TrainingLosses,
            ),
            "train_metric_manager": (
                SerializableObjectSnapshotter(self.client),
                MetricManager,
            ),
        }

        if snapshot_dir is not None:
            self.checkpointer = PerRoundStateCheckpointer(snapshot_dir)
            self.checkpoint_name = f"temp_{self.client.client_name}.pt"

    def add_default_snapshot_attr(
        self, name: str, snapshot_class: Callable[[BasicClient], AbstractSnapshotter], input_type: type[T]
    ) -> None:
        self.snapshot_attrs.update({name: (snapshot_class(self.client), input_type)})

    def delete_default_snapshot_attr(self, name: str) -> None:
        del self.snapshot_attrs[name]

    def save_snapshot(self) -> None:
        """
        Creates a snapshot of the client state and if snapshot_ckpt is given, saves it to the checkpoint.
        """
        for attr, (snapshotter_function, expected_type) in self.snapshot_attrs.items():
            self.snapshot_ckpt.update(snapshotter_function.save(attr, expected_type))

        if self.checkpointer is not None:
            self.checkpointer.save_checkpoint(self.checkpoint_name, self.snapshot_ckpt)
            self.snapshot_ckpt.clear()

        log(
            INFO,
            f"Saving client best state to checkpoint at {self.checkpointer.checkpoint_dir}"
            "with name temp_{self.client.client_name}.pt",
        )

    def load_snapshot(self, attrs: list[str] | None = None) -> None:
        """
        Load checkpointed snapshot dict consisting to the respective model attributes.

        Args:
            args (list[str] | None): List of attributes to load from the checkpoint.
                If None, all attributes are loaded. Defaults to None.
        """
        assert (
            self.checkpointer.checkpoint_exists(self.checkpoint_name) or self.snapshot_ckpt != {}
        ), "No checkpoint to load"

        if attrs is None:
            attrs = list(self.snapshot_attrs.keys())

        if self.checkpointer.checkpoint_exists(self.checkpoint_name):
            self.snapshot_ckpt = self.checkpointer.load_checkpoint(self.checkpoint_name)

        for attr in attrs:
            snapshotter, expected_type = self.snapshot_attrs[attr]
            snapshotter.load(self.snapshot_ckpt, attr, expected_type)

    def should_stop(self) -> bool:
        """
        Determine if the client should stop training based on early stopping criteria.

        Returns:
            bool: True if training should stop, otherwise False.
        """

        val_loss, _ = self.client._validate_or_test(
            loader=self.client.val_loader,
            loss_meter=self.client.val_loss_meter,
            metric_manager=self.client.val_metric_manager,
            logging_mode=LoggingMode.EARLY_STOP_VALIDATION,
            include_losses_in_metrics=False,
        )

        if val_loss is None:
            return False

        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.count_down = self.patience
            self.save_snapshot()
            return False

        if self.count_down is not None:
            self.count_down -= 1
            if self.count_down <= 0:
                return True

        return False
