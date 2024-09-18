import os
from abc import ABC, abstractmethod
from logging import INFO
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

CheckpointScoreFunctionType = Callable[[float, Dict[str, Scalar]], float]


class TorchCheckpointer(ABC):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        Basic abstract base class to handle checkpointing pytorch models. Models are saved with torch.save by default

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
        """
        self.best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    @abstractmethod
    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: Dict[str, Scalar]) -> None:
        """
        Abstract method to be implemented by every TorchCheckpointer. Based on the loss and metrics provided it should
        determine whether to produce a checkpoint AND save it if applicable.

        Args:
            model (nn.Module): Model to potentially save via the checkpointer
            loss (float): Computed loss associated with the model.
            metrics (Dict[str, float]): Computed metrics associated with the model.

        Raises:
            NotImplementedError: Must be implemented by the checkpointer
        """
        raise NotImplementedError

    def load_best_checkpoint(self) -> nn.Module:
        return torch.load(self.best_checkpoint_path)


class FunctionTorchCheckpointer(TorchCheckpointer):
    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_name: str,
        checkpoint_score_function: CheckpointScoreFunctionType,
        maximize: bool = False,
    ) -> None:
        """
        A general torch checkpointer base class that allows for flexible definition of how to decide when to checkpoint
        based on the loss and metrics provided. The score function should compute a score from these values and
        maximize specifies whether we are hoping to maximize or minimize that score

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
            checkpoint_score_function (CheckpointFunctionType): Function taking in a loss value and dictionary of
                metrics and produces a score based on these.
            maximize (bool, optional): Specifies whether we're trying to minimize or maximize the score produced
                by the scoring function. Defaults to False.
        """
        super().__init__(checkpoint_dir, checkpoint_name)
        self.best_score: Optional[float] = None
        self.checkpoint_score_function = checkpoint_score_function
        # Whether we're looking to maximize (or minimize) the score produced by the checkpoint score function
        self.maximize = maximize
        self.comparison_str = ">=" if self.maximize else "<="

    def _should_checkpoint(self, comparison_score: float) -> bool:
        # Compares the current score to the best previously recorded, returns true if should checkpoint and false
        # otherwise
        if self.best_score:
            if self.maximize:
                return self.best_score <= comparison_score
            else:
                return self.best_score >= comparison_score

        # If best score is none, then this is the first checkpoint
        return True

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: Dict[str, Scalar]) -> None:
        # First we use the provided scoring function to produce a score
        comparison_score = self.checkpoint_score_function(loss, metrics)
        if self._should_checkpoint(comparison_score):
            log(
                INFO,
                f"Checkpointing the model: Current score ({comparison_score}) "
                f"{self.comparison_str} Best score ({self.best_score})",
            )
            self.best_score = comparison_score
            torch.save(model, self.best_checkpoint_path)
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current score ({comparison_score}) is not "
                f"{self.comparison_str} Best score ({self.best_score})",
            )


class LatestTorchCheckpointer(FunctionTorchCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        # This function is required by the parent class, but not used in the LatestTorchCheckpointer
        def null_score_function(loss: float, _: Dict[str, Scalar]) -> float:
            return 0.0

        super().__init__(checkpoint_dir, checkpoint_name, null_score_function, False)

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: Dict[str, Scalar]) -> None:
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        torch.save(model, self.best_checkpoint_path)


class BestLossTorchCheckpointer(FunctionTorchCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        This checkpointer only uses the loss value provided to the maybe_checkpoint function to determine whether a
        checkpoint should be save. We are always attempting to minimize the loss. So maximize is always set to false

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
        """

        # The BestLossTorchCheckpointer just uses the provided loss to scoring checkpoints. More complicated
        # approaches may be used by other classes.
        def loss_score_function(loss: float, _: Dict[str, Scalar]) -> float:
            return loss

        super().__init__(
            checkpoint_dir, checkpoint_name, checkpoint_score_function=loss_score_function, maximize=False
        )

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: Dict[str, Scalar]) -> None:
        # First we use the provided scoring function to produce a score
        comparison_score = self.checkpoint_score_function(loss, metrics)
        if self._should_checkpoint(comparison_score):
            log(
                INFO,
                f"Checkpointing the model: Current Loss ({comparison_score}) "
                f"{self.comparison_str} Best Loss ({self.best_score})",
            )
            self.best_score = comparison_score
            torch.save(model, self.best_checkpoint_path)
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current Loss ({comparison_score}) is not "
                f"{self.comparison_str} Best Loss ({self.best_score})",
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

    def save_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> None:
        """
        Saves checkpoint_dict to checkpoint path.

        Args:
            checkpoint_dict (Dict[str, Any]): A dictionary with string keys and values of type
                Any representing the state to checkpoint.
        """
        torch.save(checkpoint_dict, self.checkpoint_path)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Loads and returns the most recent checkpoint if it exists.

        Returns:
            Dict[str, Any] A dictionary representing the checkpointed state.
        """
        assert self.checkpoint_exists()

        return torch.load(self.checkpoint_path)

    def checkpoint_exists(self) -> bool:
        """
        Checks if a checkpoint exists at the checkpoint_path constructed at initialization.

        Returns:
            bool: Whether or not a checkpoint exists.
        """
        return os.path.exists(self.checkpoint_path)
