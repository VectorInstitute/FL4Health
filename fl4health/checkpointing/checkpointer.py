import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

CheckpointScoreFunctionType = Callable[[float, dict[str, Scalar]], float]


class TorchModuleCheckpointer(ABC):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        Basic abstract base class to handle checkpointing pytorch models. Models are saved with torch.save by default

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
        """
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    @abstractmethod
    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: dict[str, Scalar]) -> None:
        """
        Abstract method to be implemented by every TorchCheckpointer. Based on the loss and metrics provided it should
        determine whether to produce a checkpoint AND save it if applicable.

        Args:
            model (nn.Module): Model to potentially save via the checkpointer
            loss (float): Computed loss associated with the model.
            metrics (dict[str, float]): Computed metrics associated with the model.

        Raises:
            NotImplementedError: Must be implemented by the checkpointer
        """
        raise NotImplementedError("maybe_checkpoint must be implemented by inheriting classes")

    def load_checkpoint(self, path_to_checkpoint: str | None = None) -> nn.Module:
        """
        Checkpointer with the option to either specify a checkpoint path or fall back on the internal path of the
        checkpointer. The flexibility to specify a load path is useful, for example, if you are not overwriting
        checkpoints when saving and need to load a specific past checkpoint for whatever reason.

        Args:
            path_to_checkpoint (str | None, optional): If provided, the checkpoint will be loaded from this path.
                If not specified, the checkpointer will load from self.checkpoint_path. Defaults to None.

        Returns:
            nn.Module: Returns a torch module loaded from the proper checkpoint path.
        """
        if path_to_checkpoint is None:
            return torch.load(self.checkpoint_path)
        return torch.load(path_to_checkpoint)


class FunctionTorchModuleCheckpointer(TorchModuleCheckpointer):
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
            checkpoint_score_function (CheckpointScoreFunctionType): Function taking in a loss value and dictionary of
                metrics and produces a score based on these.
            maximize (bool, optional): Specifies whether we're trying to minimize or maximize the score produced
                by the scoring function. Defaults to False.
        """
        super().__init__(checkpoint_dir, checkpoint_name)
        self.best_score: float | None = None
        self.checkpoint_score_function = checkpoint_score_function
        # Whether we're looking to maximize (or minimize) the score produced by the checkpoint score function
        self.maximize = maximize
        self.comparison_str = ">=" if self.maximize else "<="

    def _should_checkpoint(self, comparison_score: float) -> bool:
        """
        Compares the current score to the best previously recorded, returns true if should checkpoint and false
        otherwise. If the previous best score is None, then we always checkpoint.

        Args:
            comparison_score (float): Score that is being maximized or minimized. Will be compared against the previous
                best score seen by this checkpointer.

        Returns:
            bool: Whether or not to checkpoint the model based on the provided score
        """

        if self.best_score:
            if self.maximize:
                return self.best_score <= comparison_score
            return self.best_score >= comparison_score

        # If best score is none, then this is the first checkpoint
        return True

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: dict[str, Scalar]) -> None:
        """
        Given the loss/metrics associated with the provided model, the checkpointer uses the scoring function to
        produce a score. This score will then be used to determine whether the model should be checkpointed or not.

        Args:
            model (nn.Module): Model that might be persisted if the scoring function determines it should be
            loss (float): Loss associated with the provided model. Will potentially contribute to checkpointing
                decision, based on the score function.
            metrics (dict[str, Scalar]): Metrics associated with the provided model. Will potentially contribute to
                the checkpointing decision, based on the score function.

        Raises:
            e: Will throw an error if there is an issue saving the model. Torch.save seems to swallow errors in this
                context, so we explicitly surface the error with a try/except.
        """
        # First we use the provided scoring function to produce a score
        comparison_score = self.checkpoint_score_function(loss, metrics)
        if self._should_checkpoint(comparison_score):
            log(
                INFO,
                f"Checkpointing the model: Current score ({comparison_score}) "
                f"{self.comparison_str} Best score ({self.best_score})",
            )
            self.best_score = comparison_score
            try:
                log(INFO, f"Saving checkpoint as {str(self.checkpoint_path)}")
                torch.save(model, self.checkpoint_path)
            except Exception as e:
                log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
                raise e
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current score ({comparison_score}) is not "
                f"{self.comparison_str} Best score ({self.best_score})",
            )


class LatestTorchModuleCheckpointer(FunctionTorchModuleCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        A checkpointer that always checkpoints the model, regardless of the loss/metrics provided. As such, the score
        function is essentially a dummy.

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
        """

        # This function is required by the parent class, but not used in the LatestTorchCheckpointer
        def null_score_function(loss: float, _: dict[str, Scalar]) -> float:
            return 0.0

        super().__init__(checkpoint_dir, checkpoint_name, null_score_function, False)

    def maybe_checkpoint(self, model: nn.Module, loss: float, _: dict[str, Scalar]) -> None:
        """
        This function is essentially a pass through, as this class always checkpoints the provided model

        Args:
            model (nn.Module): Model to be checkpointed whenever this function is called
            loss (float): Loss associated with the provided model. Will potentially contribute to checkpointing
                decision, based on the score function. NOT USED.
            metrics (dict[str, Scalar]): Metrics associated with the provided model. Will potentially contribute to
                the checkpointing decision, based on the score function. NOT USED.

        Raises:
            e: Will throw an error if there is an issue saving the model. Torch.save seems to swallow errors in this
                context, so we explicitly surface the error with a try/except.
        """
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        try:
            log(INFO, f"Saving checkpoint as {str(self.checkpoint_path)}")
            torch.save(model, self.checkpoint_path)
        except Exception as e:
            log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
            raise e


class BestLossTorchModuleCheckpointer(FunctionTorchModuleCheckpointer):
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
        def loss_score_function(loss: float, _: dict[str, Scalar]) -> float:
            return loss

        super().__init__(
            checkpoint_dir, checkpoint_name, checkpoint_score_function=loss_score_function, maximize=False
        )

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: dict[str, Scalar]) -> None:
        """
        This function will decide whether to checkpoint the provided model based on the loss argument. If the provided
        loss is better than any previous losses seen by this checkpointer, the model will be saved.

        Args:
            model (nn.Module): Model that might be persisted if the scoring function determines it should be
            loss (float): Loss associated with the provided model. This value is used to determine whether to save the
                model or not.
            metrics (dict[str, Scalar]): Metrics associated with the provided model. Will not be used by this
                checkpointer.

        Raises:
            e: Will throw an error if there is an issue saving the model. Torch.save seems to swallow errors in this
                context, so we explicitly surface the error with a try/except.
        """
        # First we use the provided scoring function to produce a score
        comparison_score = self.checkpoint_score_function(loss, metrics)
        if self._should_checkpoint(comparison_score):
            log(
                INFO,
                f"Checkpointing the model: Current Loss ({comparison_score}) "
                f"{self.comparison_str} Best Loss ({self.best_score})",
            )
            self.best_score = comparison_score
            try:
                log(INFO, f"Saving checkpoint as {str(self.checkpoint_path)}")
                torch.save(model, self.checkpoint_path)
            except Exception as e:
                log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
                raise e
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current Loss ({comparison_score}) is not "
                f"{self.comparison_str} Best Loss ({self.best_score})",
            )


class PerRoundStateCheckpointer:
    def __init__(self, checkpoint_dir: Path) -> None:
        """
        Base class that provides a uniform interface for loading, saving and checking if checkpoints exists.

        Args:
            checkpoint_dir (Path): Base directory to store checkpoints. This checkpoint directory MUST already exist.
            It will not be created by this state checkpointer.
        """
        log(
            WARNING,
            "Creating PerRoundCheckpointer. Currently, this functionality is still experimental and only supported "
            "for BasicClient and NnunetClient, along with their associated servers.",
        )
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, checkpoint_name: str, checkpoint_dict: dict[str, Any]) -> None:
        """
        Saves checkpoint_dict to checkpoint path form from this classes checkpointer dir and the provided checkpoint
        name.


        Args:
            checkpoint_name (str): Name of the state checkpoint file.
            checkpoint_dict (dict[str, Any]): A dictionary with string keys and values of type Any representing the
                state to checkpoint.

        Raises:
            e: Will throw an error if there is an issue saving the model. Torch.save seems to swallow errors in this
                context, so we explicitly surface the error with a try/except.
        """

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        try:
            log(INFO, f"Saving state as {checkpoint_path}")
            torch.save(checkpoint_dict, checkpoint_path)
        except Exception as e:
            log(ERROR, f"Encountered the following error while saving the checkpoint: {e}")
            raise e

    def load_checkpoint(self, checkpoint_name: str) -> dict[str, Any]:
        """
        Loads and returns the checkpoint stored in checkpoint_dir under the provided name if it exists.
        If it doesn't exist, an assertion error will be thrown.

        Args:
            checkpoint_name (str): Name of the state checkpoint to be loaded.

        Returns:
            dict[str, Any]: A dictionary representing the checkpointed state, as loaded by torch.load.
        """

        assert self.checkpoint_exists(checkpoint_name)
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        log(INFO, f"Loading state from checkpoint at {checkpoint_path}")

        return torch.load(checkpoint_path)

    def checkpoint_exists(self, checkpoint_name: str, **kwargs: Any) -> bool:
        """
        Checks if a checkpoint exists at the checkpoint_dir constructed at initialization + checkpoint_name.

        Returns:
            bool: Whether or not a checkpoint exists.
        """
        if "checkpoint_path" in kwargs:
            raise ValueError(
                "Previously this checkpoint supported sending a path, but it now requires a checkpoint_name only"
            )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        return os.path.exists(checkpoint_path)
