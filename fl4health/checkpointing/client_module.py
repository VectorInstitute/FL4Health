from collections.abc import Sequence
from enum import Enum
from logging import INFO
from typing import Any, Union

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import PerRoundStateCheckpointer, TorchModuleCheckpointer

CheckpointModuleInput = Union[TorchModuleCheckpointer, Sequence[TorchModuleCheckpointer]] | None


class CheckpointMode(Enum):
    PRE_AGGREGATION = "pre_aggregation"
    POST_AGGREGATION = "post_aggregation"


class ClientCheckpointAndStateModule:
    def __init__(
        self,
        pre_aggregation: CheckpointModuleInput = None,
        post_aggregation: CheckpointModuleInput = None,
        state_checkpointer: PerRoundStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to hold up three major components that determine how clients handle model and state
        checkpointing, where state checkpointing is meant to allow clients to restart if FL training is interrupted.
        For model checkpointing, there are two distinct types.
            The first type, if defined, is used to checkpoint local models BEFORE server-side aggregation, but
            after local training. **NOTE**: This is akin to "further fine-tuning" approaches for global models.

            The second type, if defined, is used to checkpoint local models AFTER server-side aggregation, but
            before local training **NOTE**: This is the "traditional" mechanism for global models.

        As a final note, for some methods, such as Ditto or MR-MTL, these checkpoints will actually be identical.
        That's because the target model for these methods is never globally aggregated. That is, they remain local

        Args:
            pre_aggregation (CheckpointModuleInput, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their validation metrics/losses **BEFORE**
                server-side aggregation. Defaults to None.
            post_aggregation (CheckpointModuleInput, optional): If defined, this checkpointer (or sequence
                of checkpointers) is used to checkpoint models based on their validation metrics/losses **AFTER**
                server-side aggregation. Defaults to None.
            state_checkpointer (PerRoundStateCheckpointer | None, optional): If defined, this checkpointer is used to
                preserve client state (not just models), in the event one wants to restart federated training.
                Defaults to None.
        """
        self.pre_aggregation = (
            [pre_aggregation] if isinstance(pre_aggregation, TorchModuleCheckpointer) else pre_aggregation
        )
        self.post_aggregation = (
            [post_aggregation] if isinstance(post_aggregation, TorchModuleCheckpointer) else post_aggregation
        )
        self._check_if_shared_checkpoint_names()
        self.state_checkpointer = state_checkpointer

    def _check_if_shared_checkpoint_names(self) -> None:
        """
        This function checks whether there is overlap in the paths to which the checkpointers of this module are
        supposed to write. This is to ensure that there isn't any accidental overwriting of checkpoints that is
        unintended.

        Raises:
            ValueError: If any of the pre- or post-aggregation model checkpointer paths are not unique.
        """

        pre_aggregation_paths = (
            [checkpointer.checkpoint_path for checkpointer in self.pre_aggregation] if self.pre_aggregation else []
        )
        post_aggregation_paths = (
            [checkpointer.checkpoint_path for checkpointer in self.post_aggregation] if self.post_aggregation else []
        )

        all_paths = pre_aggregation_paths + post_aggregation_paths
        unique_paths = set(all_paths)

        if len(unique_paths) != len(all_paths):
            formatted_all_paths = "\n".join(all_paths)
            raise ValueError(
                "The paths of all of your checkpointers should be unique otherwise overwrites are possible and data "
                f"will be lost. The current paths are:\n{formatted_all_paths}"
            )

    def maybe_checkpoint(
        self, model: nn.Module, loss: float, metrics: dict[str, Scalar], mode: CheckpointMode
    ) -> None:
        """
        Performs model checkpointing for a particular mode (either pre- or post-aggregation) if any checkpointers are
        provided for that particular mode in this module. If present, the various checkpointers will decide whether
        or not to checkpoint based on their internal criterion and the loss/metrics provided.

        Args:
            model (nn.Module): The model that might be checkpointed by the checkpointers.
            loss (float): The metric value obtained by the provided model. Used by the checkpointer(s) to decide
                whether to checkpoint the model.
            metrics (dict[str, Scalar]): The metrics obtained by the provided model. Potentially used by checkpointer
                to decide whether to checkpoint the model.
            mode (CheckpointMode): Determines which of the types of checkpointers to use. Currently, the only modes
                available are pre- and post-aggregation.

        Raises:
            ValueError: Thrown if the model checkpointing mode is not recognized.
        """
        if mode == CheckpointMode.PRE_AGGREGATION:
            if self.pre_aggregation is not None:
                for checkpointer in self.pre_aggregation:
                    checkpointer.maybe_checkpoint(model, loss, metrics)
            else:
                log(INFO, "No Pre-aggregation checkpoint specified. Skipping.")
        elif mode == CheckpointMode.POST_AGGREGATION:
            if self.post_aggregation is not None:
                for checkpointer in self.post_aggregation:
                    checkpointer.maybe_checkpoint(model, loss, metrics)
            else:
                log(INFO, "No Post-aggregation checkpoint specified. Skipping.")
        else:
            raise ValueError(f"Unrecognized mode for checkpointing: {str(mode)}")

    def save_state(self, state_checkpoint_name: str, state: dict[str, Any]) -> None:
        """
        This function is meant to facilitate saving state required to restart an FL process on the client side. This
        function will simply save whatever information is passed in the state variable using the file name in
        state_checkpoint_name. This function should only be called if a state_checkpointer exists in this module

        Args:
            state_checkpoint_name (str): Name of the state checkpoint file. The checkpointer itself will have a
                directory to which state will be saved.
            state (dict[str, Any]): State to be saved so that training might be resumed on the client if federated
                training is interrupted. For example, this might contain things like optimizer states, learning rate
                scheduler states, etc.

        Raises:
            ValueError: Throws an error if this function is called, but no state checkpointer has been provided
        """

        if self.state_checkpointer is not None:
            self.state_checkpointer.save_checkpoint(state_checkpoint_name, state)
        else:
            raise ValueError("Attempting to save state but no state checkpointer is specified")

    def maybe_load_state(self, state_checkpoint_name: str) -> dict[str, Any] | None:
        """
        This function facilitates loading of any pre-existing state (with the name state_checkpoint_name) in the
        directory of the state_checkpointer. If the state already exists at the proper path, the state is loaded
        and returned. If it doesn't exist, we return None.

        Args:
            state_checkpoint_name (str): Name of the state checkpoint file. The checkpointer itself will have a
                directory from which state will be loaded (if it exists).

        Raises:
            ValueError: Throws an error if this function is called, but no state checkpointer has been provided

        Returns:
            dict[str, Any] | None: If the state checkpoint properly exists and is loaded correctly, this dictionary
                carries that state. Otherwise, we return a None (or throw an exception).
        """

        if self.state_checkpointer is not None:
            if self.state_checkpointer.checkpoint_exists(state_checkpoint_name):
                return self.state_checkpointer.load_checkpoint(state_checkpoint_name)
            else:
                log(INFO, "State checkpointer is defined but no state checkpoint exists.")
                return None
        else:
            raise ValueError("Attempting to load state, but no state checkpointer is specified")
