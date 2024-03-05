import os
from abc import ABC, abstractmethod
from logging import INFO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.server.history import History
from torch.optim import Optimizer


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


class PerRoundCheckpointer(ABC):
    def __init__(self, checkpoint_dir: Path, checkpoint_name: Path) -> None:
        """
        Abstract Base Class that provides a uniform interface for loading, saving and checking
        if checkpoints exists.

        Args:
            checkpoint_dir (Path): Base directory to store checkpoints.
            checkpoint_name (Path): The file name in which to save the checkpoint.
        """
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    @abstractmethod
    def save_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> None:
        """
        Saves checkpoint_dict to checkpoint path.

        Args:
            checkpoint_dict (Dict[str, Any]): A dictionary with string keys and values of type
                Any representing the state to checkpoint.

        Raises:
            NotImplementedError: To be implemented by child classes.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self) -> Tuple[Any, ...]:
        """
        Loads and returns the most recent checkpoint if it exists.

        Returns:
            Tuple[Any, ...]: A tuple where each entry is an element of the checkpointed state.

        Raises:
            NotImplementedError: To be implemented by child classes.
        """
        raise NotImplementedError

    def checkpoint_exists(self) -> bool:
        """
        Checks if a checkpoint exists at the checkpoint_path constructed at initialization.

        Returns:
            bool: Whether or not a checkpoint exists.
        """
        return os.path.exists(self.checkpoint_path)


class CentralPerRoundCheckpointer(PerRoundCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, Optimizer, int]]) -> None:
        """
        Saves checkpoint_dict consisting of model, optimizer and round to checkpoint path.

        Args:
            checkpoint_dict (Dict[str, Union[nn.Module, Optimizer, int]]): A dictionary with string keys and values of
                type nn.Module (model), Optimizer (optimizer) and int (round).
        """
        assert "epoch" in checkpoint_dict and isinstance(checkpoint_dict["epoch"], int)
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "optimizer" in checkpoint_dict and isinstance(checkpoint_dict["optimizer"], Optimizer)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, Optimizer, int]:
        """
        Loads and returns the most recent checkpoint if it exists.

        Returns:
            Tuple[nn.Module, Optimizer, int] A tuple consisting of the model, optimizer and round.
        """
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["optimizer"], ckpt["epoch"]


class ClientPerRoundCheckpointer(PerRoundCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, Dict[str, Optimizer], str]]) -> None:
        """
        Saves checkpoint_dict consisting of model, optimizer and client name.

        Args:
            checkpoint_dict (Dict[str, Union[nn.Module, Dict[str, Optimizer], str]]): A dictionary with string keys
                and values of type nn.Module (model), a dictionary of optimizers indexed by string keys and a
                string representing the client name.
        """
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "optimizers" in checkpoint_dict and isinstance(checkpoint_dict["optimizers"], dict)
        assert "client_name" in checkpoint_dict and isinstance(checkpoint_dict["client_name"], str)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, Dict[str, Optimizer], str]:
        """
        Loads and returns the most recent checkpoint if it exists.

        Returns:
            Tuple[nn.Module, Dict[str, Optimizer], str] A tuple consisting of the model, the dictionary of optimizers
            and the client name
        """
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["optimizers"], ckpt["client_name"]


class ServerPerRoundCheckpointer(PerRoundCheckpointer):
    def save_checkpoint(self, checkpoint_dict: Dict[str, Union[nn.Module, History, int]]) -> None:
        """
        Saves checkpoint_dict consisting of model, a history of losses and metrics through validation and the server
        round.

        Args:
            checkpoint_dict (Dict[str, Union[nn.Module, History, int]]): A dictionary with string keys and values of
                type nn.Module (model), History (losses and metrics) and int (server round).
        """
        assert "model" in checkpoint_dict and isinstance(checkpoint_dict["model"], nn.Module)
        assert "history" in checkpoint_dict and isinstance(checkpoint_dict["history"], History)
        assert "server_round" in checkpoint_dict and isinstance(checkpoint_dict["server_round"], int)

        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Tuple[nn.Module, History, int]:
        """
        Loads and returns the most recent checkpoint if it exists.

        Returns:
            Tuple[nn.Module, History, int] A tuple consisting of the model, history and round for the server.
        """
        assert self.checkpoint_exists()

        ckpt = torch.load(self.checkpoint_path)
        return ckpt["model"], ckpt["history"], ckpt["server_round"]
