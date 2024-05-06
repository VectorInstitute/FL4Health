import pickle
from logging import INFO
from typing import Any, Dict

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar
from opacus import GradSampleModule

from fl4health.checkpointing.checkpointer import FunctionTorchCheckpointer


class OpacusCheckpointer(FunctionTorchCheckpointer):
    """
    This is a specific type of checkpointer to be used in saving models trained using Opacus for differential privacy.
    Certain layers within Opacus wrapped models do not interact well with torch.save functionality. This checkpointer
    fixes this issue.
    """

    def maybe_checkpoint(self, model: GradSampleModule, loss: float, metrics: Dict[str, Scalar]) -> None:
        """
        Overriding the checkpointing strategy of the FunctionTorchCheckpointer to save model state dictionaries
        instead of using the torch.save workflow.

        Args:
            model (nn.Module): Model to be potentially saved (should be an Opacus wrapped model)
            loss (float): Loss value associated with the model to be used in checkpointing decisions.
            metrics (Dict[str, Scalar]): Metrics associated with the model to be used in checkpointing decisions.
        """
        assert isinstance(
            model, GradSampleModule
        ), f"Model is of type: {type(model)}. This checkpointer need only be used to checkpoint Opacus modules"
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

    def _process_state_dict_keys(self, opacus_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        State dictionary keys for Opacus modules will be prefixed with an _module. So we remove these when loading
        the state information into a standard torch model.

        Args:
            opacus_state_dict (Dict[str, Any]): A state dictionary produced by an Opacus GradSamplingModule

        Returns:
            Dict[str, Any]: A state dictionary with the _module. removed from the key prefixes to facilitate loading
                the state dictionary into a non-Opacus model.
        """

        return {key.removeprefix("_module."): val for key, val in opacus_state_dict.items()}

    def _extract_and_save_state(self, model: nn.Module) -> None:
        """
        Certain Opacus layers don't integrate nicely with the torch.save mechanism. So rather than using that approach
        for checkpointing Opacus models, we extract and save the model state dictionary.

        Args:
            model (nn.Module): Model to be checkpointed via the state dictionary.
        """
        model_state_dict = model.state_dict()
        with open(self.best_checkpoint_path, "wb") as handle:
            pickle.dump(model_state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_best_checkpoint(self) -> nn.Module:
        raise NotImplementedError(
            "When loading from Opacus checkpointers, you need to provide a model into which state is loaded. "
            "Please use load_best_checkpoint_into_model instead and provide model architecture to load state into."
        )

    def load_best_checkpoint_into_model(
        self, target_model: nn.Module, target_is_grad_sample_module: bool = False
    ) -> None:
        """
        State dictionary loading requires a model to be provided (unlike the torch.save mechanism). So we define this
        function, which requires the user to provide a model into which the state dictionary is to be loaded.

        Args:
            target_model (nn.Module): Target model for loading state into.
            target_is_grad_sample_module (bool, optional): Whether the target_model that the state_dict is being
                loaded into is an Opacus module or just a vanilla Pytorch module. Defaults to False.
        """
        with open(self.best_checkpoint_path, "rb") as handle:
            model_state_dict = pickle.load(handle)
            # If the target is just a plain PyTorch module, we remove the _module key prefix that Opacus inserts into
            # its GradSampleModules.
            if not target_is_grad_sample_module:
                model_state_dict = self._process_state_dict_keys(model_state_dict)
            target_model.load_state_dict(model_state_dict, strict=True)


class LatestOpacusCheckpointer(OpacusCheckpointer):
    def __init__(self, checkpoint_dir: str, checkpoint_name: str) -> None:
        """
        This class implements a checkpointer that always saves the model state when called. It uses a placeholder
        scoring function and maximize argument.

        Args:
            checkpoint_dir (str): Directory to save checkpoint state to
            checkpoint_name (str): Name of the file to which state is to be saved to.
        """

        # This function is required by the parent class, but not used in the LatestOpacusCheckpointer
        def latest_score_function(loss: float, _: Dict[str, Scalar]) -> float:
            return 0.0

        super().__init__(checkpoint_dir, checkpoint_name, latest_score_function, False)

    def maybe_checkpoint(self, model: GradSampleModule, loss: float, _: Dict[str, Scalar]) -> None:
        assert isinstance(
            model, GradSampleModule
        ), f"Model is of type: {type(model)}. This checkpointer need only be used to checkpoint Opacus modules"
        # Always checkpoint the latest model
        log(INFO, "Saving latest checkpoint with LatestTorchCheckpointer")
        self._extract_and_save_state(model)


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

        # The BestLossOpacusCheckpointer just uses the provided loss to scoring checkpoints. More complicated
        # approaches may be used by other classes.
        def loss_score_function(loss: float, _: Dict[str, Scalar]) -> float:
            return loss

        super().__init__(
            checkpoint_dir, checkpoint_name, checkpoint_score_function=loss_score_function, maximize=False
        )

    def maybe_checkpoint(self, model: GradSampleModule, loss: float, metrics: Dict[str, Scalar]) -> None:
        assert isinstance(
            model, GradSampleModule
        ), f"Model is of type: {type(model)}. This checkpointer need only be used to checkpoint Opacus modules"
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
