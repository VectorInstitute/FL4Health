import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from logging import INFO
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

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

T = TypeVar("T")
# check what is different between union and this
Serizable = TypeVar("Serizable", MetricManager, TrainingLosses, ReportsManager)
Number = TypeVar("Number", int, float)


class Snapshotter(ABC, Generic[T]):
    def __init__(self, client: BasicClient) -> None:
        self.client = client

    @abstractmethod
    def save(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def load(self, ckpt: dict[str, Any], name: str) -> None:
        raise NotImplementedError


class OptimizerSnapshotter(Snapshotter[Optimizer]):

    def _get_optimizer_state(self, optimizers: Any) -> dict:
        """
        Helper function to handle extraction of optimizer states from either a single optimizer
        or a dictionary of optimizers.
        """
        if isinstance(optimizers, Optimizer):
            return {None: optimizers.state_dict()["state"]}  # Return as dict with None as key for single optimizer
        elif isinstance(optimizers, dict):
            output = {}
            for key, optimizer in optimizers.items():
                if not isinstance(optimizer, Optimizer):
                    raise ValueError(f"Unrecognized type of optimizer {type(optimizer)} for key {key}")
                output[key] = optimizer.state_dict()["state"]
            return output
        else:
            raise ValueError(f"Unrecognized type of optimizer {type(optimizers)}")

    def save(self, name: str) -> Any:
        """
        Save the state of the optimizers (either single or dictionary of them).
        """
        optimizers = getattr(self.client, name)
        return self._get_optimizer_state(optimizers)

    def _apply_optimizer_state(self, optimizers: Any, optimizer_states: dict) -> None:
        """
        Helper function to load the optimizer states into either a single optimizer
        or a dictionary of optimizers.
        """
        if isinstance(optimizers, Optimizer):
            optimizer_state_dict = optimizers.state_dict()
            optimizer_state_dict["state"] = optimizer_states.get(None)
            optimizers.load_state_dict(optimizer_state_dict)
        elif isinstance(optimizers, dict):
            for key, optimizer in optimizers.items():
                optimizer_state_dict = optimizer.state_dict()
                optimizer_state_dict["state"] = optimizer_states[key]
                optimizer.load_state_dict(optimizer_state_dict)
        else:
            raise ValueError(f"Unrecognized type of optimizer {type(optimizers)}")

    def load(self, ckpt: dict[str, Any], name: str) -> None:
        """
        Load the optimizer states (either single or dictionary of them) from the checkpoint.
        """
        optimizers = getattr(self.client, name)
        optimizer_states = ckpt[name]
        self._apply_optimizer_state(optimizers, optimizer_states)


class LRSchedulerSnapshotter(Snapshotter[_LRScheduler]):

    def _get_lr_scheduler_state(self, lr_schedulers: Any) -> dict:
        """
        Helper function to handle extraction of optimizer states from either a single LR scheduler
        or a dictionary of LR schedulers.
        """
        if isinstance(lr_schedulers, _LRScheduler):
            return {None: lr_schedulers.state_dict()}  # Return as dict with None as key for single optimizer
        elif isinstance(lr_schedulers, dict):
            output = {}
            for key, lr_scheduler in lr_schedulers.items():
                if not isinstance(lr_scheduler, _LRScheduler):
                    raise ValueError(f"Unrecognized type of LR scheduler {type(lr_scheduler)} for key {key}")
                output[key] = lr_scheduler.state_dict()["state"]
            return output
        else:
            raise ValueError(f"Unrecognized type of LR scheduler {type(lr_schedulers)}")

    def save(self, name: str) -> Any:
        """
        Save the state of the LR scheduler (either single or dictionary of them).
        """
        lr_schedulers = getattr(self.client, name)
        return self._get_lr_scheduler_state(lr_schedulers)

    def _apply_lr_scheduler_state(self, lr_schedulers: Any, lr_scheduler_states: dict) -> None:
        """
        Helper function to load the LR scheduler states into either a single LR scheduler
        or a dictionary of LR schedulers.
        """
        if isinstance(lr_schedulers, _LRScheduler):
            lr_scheduler_state_dict = lr_schedulers.state_dict()
            lr_scheduler_state_dict["state"] = lr_scheduler_states.get(None)
            lr_schedulers.load_state_dict(lr_scheduler_state_dict)
        elif isinstance(lr_schedulers, dict):
            for key, lr_scheduler in lr_schedulers.items():
                lr_scheduler_state_dict = lr_scheduler.state_dict()
                lr_scheduler_state_dict["state"] = lr_scheduler_states[key]
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        else:
            raise ValueError(f"Unrecognized type of LR scheduler {type(lr_schedulers)}")

    def load(self, ckpt: dict[str, Any], name: str) -> None:
        """
        Load the LR Scheduler states (either single or dictionary of them) from the checkpoint.
        """
        lr_schedulers = getattr(self.client, name)
        lr_scheduler_states = ckpt[name]
        self._apply_lr_scheduler_state(lr_schedulers, lr_scheduler_states)


class TorchModuleSnapshotter(Snapshotter[nn.Module]):

    def _get_model_state(self, models: Any) -> dict:
        """
        Helper function to handle extraction of optimizer states from either a single model
        or a dictionary of models.
        """
        if isinstance(models, nn.Module):
            return {None: models.state_dict()}  # Return as dict with None as key for single optimizer
        elif isinstance(models, dict):
            output = {}
            for key, model in models.items():
                if not isinstance(model, nn.Module):
                    raise ValueError(f"Unrecognized type of nn.Module {type(model)} for key {key}")
                output[key] = model.state_dict()["state"]
            return output
        else:
            raise ValueError(f"Unrecognized type of nn.Module {type(models)}")

    def save(self, name: str) -> Any:
        """
        Save the state of the model (either single or dictionary of them).
        """
        models = getattr(self.client, name)
        return self._get_model_state(models)

    def _apply_model_state(self, models: Any, models_states: dict) -> None:
        """
        Helper function to load the nn.Module states into either a single model
        or a dictionary of models.
        """
        if isinstance(models, nn.Module):
            model_state_dict = models.state_dict()
            model_state_dict["state"] = models.get(None)
            models.load_state_dict(model_state_dict)
        elif isinstance(models, dict):
            for key, model in models.items():
                model_state_dict = model.state_dict()
                model_state_dict["state"] = models_states[key]
                model.load_state_dict(model_state_dict)
        else:
            raise ValueError(f"Unrecognized type of models {type(models)}")

    def load(self, ckpt: dict[str, Any], name: str) -> None:
        """
        Load the models states (either single or dictionary of them) from the checkpoint.
        """
        lr_schedulers = getattr(self.client, name)
        lr_scheduler_states = ckpt[name]
        self._apply_model_state(lr_schedulers, lr_scheduler_states)


class SerizableObjectSnapshotter(Snapshotter[Serizable]):
    def save(self, name: str) -> Any:
        serizable = getattr(self.client, name)
        assert isinstance(
            serizable, (MetricManager, TrainingLosses, ReportsManager)
        ), f"Attribute {name} is not of the expected type"
        return copy.deepcopy(serizable)

    def load(self, ckpt: dict[str, Any], name: str) -> None:
        """
        Load the models states (either single or dictionary of them) from the checkpoint.
        """
        assert isinstance(
            ckpt[name], (MetricManager, TrainingLosses, ReportsManager)
        ), f"Attribute {name} is not of the expected type"
        setattr(self.client, name, ckpt[name])


class NumberSnapshotter(Snapshotter[Number]):
    def save(self, name: str) -> Any:
        serizable = getattr(self.client, name)
        assert isinstance(serizable, (float, int)), f"Attribute {name} is not of the expected type"
        return copy.deepcopy(serizable)

    def load(self, ckpt: dict[str, Any], name: str) -> None:
        """
        Load the models states (either single or dictionary of them) from the checkpoint.
        """
        assert isinstance(ckpt[name], (float, int)), f"Attribute {name} is not of the expected type"
        setattr(self.client, name, ckpt[name])


class EarlyStopper(ABC):
    def __init__(
        self,
        client: BasicClient,
        patience: int = -1,
        interval_steps: int = 5,
        extra_args: Optional[Sequence[str]] = None,
        snapshot_dir: Optional[Path] = None,
    ) -> None:

        # check not getting double memory usage
        self.client = client

        self.patience = patience
        self.counte_down = patience
        self.interval_steps = interval_steps

        self.best_score: Optional[float] = None
        self.snapshot_ckpt: dict[str, Any] = {}

        self.default_args: Dict[str, Snapshotter] = {
            "model": TorchModuleSnapshotter(self.client),  # for nn.Module we only need state_dict
            "optimizers": OptimizerSnapshotter(self.client),  # dict of optimizers we only need state_dict
            "lr_schedulers": LRSchedulerSnapshotter(self.client),  # dict of schedulers we only need state_dict
            "learning_rate": NumberSnapshotter(self.client),  # number we can copy
            "total_steps": NumberSnapshotter(self.client),  # number we can copy
            "total_epochs": NumberSnapshotter(self.client),  # number we can copy
            "reports_manager": SerizableObjectSnapshotter(self.client),  # Class of objects we can copy
            "train_loss_meter": SerizableObjectSnapshotter(self.client),  # Class of objects we can copy
            "train_metric_manager": SerizableObjectSnapshotter(self.client),  # Class of objects we can copy
        }

        if snapshot_dir is not None:
            self.checkpointer = PerRoundCheckpointer(snapshot_dir, Path(f"temp_{self.client.client_name}.pt"))

    def save_snapshot(self) -> None:
        """
        Saves checkpoint dict consisting of client name, total steps, lr schedulers,
            metrics reporter and optimizers state. Method can be overridden to augment saved checkpointed state.
        """
        for arg, snapshotter_function in self.default_args.items():
            self.snapshot_ckpt[arg] = snapshotter_function.save(arg)

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

        for arg, snapshotter_function in self.default_args.items():
            snapshotter_function.load(self.snapshot_ckpt, arg)

    def should_stop(self) -> bool:
        """
        Check if the client should stop training based on the current state of the client.
        """
        val_loss, val_metrics = self.client._validate_or_test(
            self.client.val_loader,
            self.client.val_loss_meter,
            self.client.val_metric_manager,
            logging_mode=LoggingMode.EARLY_STOP_VALIDATION,
            include_losses_in_metrics=False,  # Todo check that
        )
        if val_loss is None:
            return False
        else:
            if self.best_score is None:
                self.best_score = val_loss
                return False
            if val_loss > self.best_score:
                self.best_score = val_loss
                self.counte_down = self.patience
                self.save_snapshot()
            else:
                self.counte_down -= 1
            if self.counte_down == 0:
                self.load_snapshot()
                return True
            else:
                return False
