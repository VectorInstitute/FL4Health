import pickle
from logging import INFO
from typing import Dict

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import FunctionTorchCheckpointer


class OpacusCheckpointer(FunctionTorchCheckpointer):
    """
    _summary_
    """

    def maybe_checkpoint(self, model: nn.Module, loss: float, metrics: Dict[str, Scalar]) -> None:
        """
        _summary_

        Args:
            model (nn.Module): _description_
            loss (float): _description_
            metrics (Dict[str, Scalar]): _description_
        """
        comparison_score = self.checkpoint_score_function(loss, metrics)
        if self._should_checkpoint(comparison_score):
            log(
                INFO,
                f"Saving Opacus model state: Current score ({comparison_score}) "
                f"{self.comparison_str} Best score ({self.best_score})",
            )
            self.best_score = comparison_score
            # Extract the state dictionary for the model and save it.
            self._extract_and_save_state(model)
        else:
            log(
                INFO,
                f"Not saving Opacus model state: Current score ({comparison_score}) is not "
                f"{self.comparison_str} Best score ({self.best_score})",
            )

    def _extract_and_save_state(self, model: nn.Module) -> None:
        """
        _summary_

        Args:
            model (nn.Module): _description_
        """
        model_state_dict = model.state_dict()
        with open(self.best_checkpoint_path, "wb") as handle:
            pickle.dump(model_state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_best_checkpoint(self) -> nn.Module:
        raise NotImplementedError(
            "When loading from Opacus checkpointers, you need to provide a model into which state is loaded. "
            "Please use load_best_checkpoint_into_model instead"
        )

    def load_best_checkpoint_into_model(self, model: nn.Module) -> nn.Module:
        """
        _summary_

        Args:
            model (nn.Module): _description_

        Returns:
            nn.Module: _description_
        """
        with open(self.best_checkpoint_path, "wb") as handle:
            model_state_dict = pickle.load(handle)
            model.load_state_dict(model_state_dict, strict=True)
            return model


def latest_score_function(loss: float, _: Dict[str, Scalar]) -> float:
    return 0.0


class LatestOpacusCheckpointer(OpacusCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        super().__init__(checkpoint_dir, checkpoint_name, latest_score_function, False)

    def maybe_checkpoint(self, model: nn.Module, loss: float, _: Dict[str, Scalar]) -> None:
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        self._extract_and_save_state(model)


def loss_score_function(loss: float, _: Dict[str, Scalar]) -> float:
    return loss


class BestLossOpacusCheckpointer(OpacusCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        This checkpointer only uses the loss value provided to the maybe_checkpoint function to determine whether a
        checkpoint should be save. We are always attempting to minimize the loss. So maximize is always set to false

        Args:
            checkpoint_dir (str): Directory to which the model is saved. This directory should already exist. The
                checkpointer will not create it if it does not.
            checkpoint_name (str): Name of the checkpoint to be saved.
        """
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
            self._extract_and_save_state(model)
        else:
            log(
                INFO,
                f"Not checkpointing the model: Current Loss ({comparison_score}) is not "
                f"{self.comparison_str} Best Loss ({self.best_score})",
            )
