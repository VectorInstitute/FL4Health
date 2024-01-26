import os
from abc import ABC, abstractmethod
from logging import INFO
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from flwr.common.logger import log
from flwr.server.history import History

from fl4health.utils.metrics import MetricManager


class TorchCheckpointer:
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str) -> None:
        self.best_checkpoint_path = os.path.join(best_checkpoint_dir, best_checkpoint_name)

    def maybe_checkpoint(self, model: nn.Module, _: Optional[float] = None) -> None:
        raise NotImplementedError

    def load_best_checkpoint(self) -> nn.Module:
        return torch.load(self.best_checkpoint_path)


class LatestTorchCheckpointer(TorchCheckpointer):
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str) -> None:
        super().__init__(best_checkpoint_dir, best_checkpoint_name)

    def maybe_checkpoint(self, model: nn.Module, _: Optional[float] = None) -> None:
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        torch.save(model, self.best_checkpoint_path)


class BestMetricTorchCheckpointer(TorchCheckpointer):
    def __init__(self, best_checkpoint_dir: str, best_checkpoint_name: str, maximize: bool = False) -> None:
        super().__init__(best_checkpoint_dir, best_checkpoint_name)
        self.best_metric: Optional[float] = None
        # Whether we're looking to maximize the metric (alternatively minimize)
        self.maximize = maximize
        self.comparison_str = ">=" if self.maximize else "<="

    def _should_checkpoint(self, comparison_metric: float) -> bool:
        # Compares the current metric to the best previously recorded, returns true if should checkpoint and false
        # otherwise
        if self.best_metric:
            if self.maximize:
                return self.best_metric <= comparison_metric
            else:
                return self.best_metric >= comparison_metric

        # If best metric is none, then this is the first checkpoint
        return True

    def maybe_checkpoint(self, model: nn.Module, comparison_metric: Optional[float] = None) -> None:
        assert comparison_metric is not None
        if self._should_checkpoint(comparison_metric):
            log(
                INFO,
                f"Checkpointing the model: Current metric ({comparison_metric}) "
                f"{self.comparison_str} Best metric ({self.best_metric})",
            )
            self.best_metric = comparison_metric
            torch.save(model, self.best_checkpoint_path)
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current metric ({comparison_metric}) is not "
                f"{self.comparison_str} Best metric ({self.best_metric})",
            )


class PerEpochCheckpointer:
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    @abstractmethod
    def save_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self) -> Tuple[Any, ...]:
        return NotImplementedError

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.checkpoint_path)


class CentralPerEpochCheckpointer(PerEpochCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, Optimizer, int]]) -> None:
        assert "epoch" in checkpoint_dict and isinstance(checkpoint_dict["epoch"], int)
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "optimizer" in checkpoint_dict and isinstance(checkpoint_dict["optimizer"], Optimizer)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, Optimizer, int]:
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["optimizer"], ckpt["epoch"]


class ClientPerEpochCheckpointer(PerEpochCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, Dict[str, Optimizer]]]) -> None:
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "optimizers" in checkpoint_dict and isinstance(checkpoint_dict["optimizers"], dict)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, Dict[str, Optimizer]]:
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["optimizers"]


class ServerPerEpochCheckpointer(PerEpochCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, History, int]]) -> None:
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "history" in checkpoint_dict and isinstance(checkpoint_dict["history"], History)
        assert "server_round" in checkpoint_dict and isinstance(checkpoint_dict["server_round"], int)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, History, int]:
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["history"], ckpt["server_round"]
