from enum import Enum
from logging import INFO
from typing import Dict, Optional, Sequence, Union

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer

CheckpointModuleInput = Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]]


class CheckpointMode(Enum):
    PRE_AGGREGATION = "pre_aggregation"
    POST_AGGREGATION = "post_aggregation"


class ClientCheckpointModule:
    def __init__(
        self, pre_aggregation: CheckpointModuleInput = None, post_aggregation: CheckpointModuleInput = None
    ) -> None:
        """
        This module is meant to hold up to two distinct client-side checkpointers.
        The first checkpointer, if defined, is used to checkpoint local models BEFORE server-side aggregation.
        **NOTE**: This is akin to "further fine-tuning" approaches for global models.
        The second checkpointer, if defined, is used to checkpoint local models AFTER server-side aggregation
        **NOTE**: This is the "traditional" mechanism for global models.

        As a final note, for some methods, such as Ditto or MR-MTL, these checkpoints will actually be identical.
        That's because the target model for these methods is never globally aggregated. That is, they remain local

        Args:
            pre_aggregation (CheckpointModuleInput, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their validation metrics/losses **BEFORE**
                server-side aggregation. Defaults to None.
            post_aggregation (CheckpointModuleInput, optional], optional): If defined, this checkpointer (or sequence
                of checkpointers) is used to checkpoint models based on their validation metrics/losses **AFTER**
                server-side aggregation. Defaults to None.
        """
        self.pre_aggregation = [pre_aggregation] if isinstance(pre_aggregation, TorchCheckpointer) else pre_aggregation
        self.post_aggregation = (
            [post_aggregation] if isinstance(post_aggregation, TorchCheckpointer) else post_aggregation
        )
        self._check_if_shared_checkpoint_names()

    def _check_if_shared_checkpoint_names(self) -> None:
        """
        This function is meant to throw an exception if there is an overlap in the paths to which their checkpointers
        will save checkpoints to avoid accidental overwriting.
        """

        pre_aggregation_paths = (
            [checkpointer.best_checkpoint_path for checkpointer in self.pre_aggregation]
            if self.pre_aggregation
            else []
        )
        post_aggregation_paths = (
            [checkpointer.best_checkpoint_path for checkpointer in self.post_aggregation]
            if self.post_aggregation
            else []
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
        self, model: nn.Module, loss: float, metrics: Dict[str, Scalar], mode: CheckpointMode
    ) -> None:
        """
        If checkpointer or checkpoints indicated by the checkpoint mode exists, maybe checkpoint model based on the
        model metrics or loss

        Args:
            loss (float): The metric value obtained by the current model.
                Used by checkpointer to decide whether to checkpoint the model.
            mode (CheckpointMode): Determines which of the checkpointers to use.

        Args:
            model (nn.Module): The model that might be checkpointed by the checkpointers.
            loss (float): The loss value obtained by the current model. Potentially used by checkpointer to decide
                whether to checkpoint the model.
            metrics (Dict[str, Scalar]): The metrics obtained by the current model. Potentially used by checkpointer
                to decide whether to checkpoint the model.
            mode (CheckpointMode): Determines which of the checkpointers to use.

        Raises:
            ValueError: Thrown if the model provided is not recognized.
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
