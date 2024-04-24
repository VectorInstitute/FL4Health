from enum import Enum
from logging import INFO
from typing import Dict, Optional

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer


class CheckpointMode(Enum):
    PRE_AGGREGATION = "pre_aggregation"
    POST_AGGREGATION = "post_aggregation"


class ClientCheckpointModule:
    def __init__(
        self, pre_aggregation: Optional[TorchCheckpointer] = None, post_aggregation: Optional[TorchCheckpointer] = None
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
            pre_aggregation (Optional[TorchCheckpointer], optional): If defined, this checkpointer is used to
                checkpoint models based on their validation metrics/losses **BEFORE** server-side aggregation.
                Defaults to None.
            post_aggregation (Optional[TorchCheckpointer], optional): If defined, this checkpointer is used to
                checkpoint models based on their validation metrics/losses **AFTER** server-side aggregation.
                Defaults to None.
        """
        self.pre_aggregation = pre_aggregation
        self.post_aggregation = post_aggregation

    def maybe_checkpoint(
        self, model: nn.Module, loss: float, metrics: Dict[str, Scalar], mode: CheckpointMode
    ) -> None:
        """
        If checkpointer exists, maybe checkpoint model based on the current comparison metric value.

        Args:
            current_metric_value (float): The metric value obtained by the current model.
                Used by checkpointer to decide whether to checkpoint the model.
            mode (CheckpointMode): Determines which of the checkpointers to use.
        """
        if mode == CheckpointMode.PRE_AGGREGATION:
            if self.pre_aggregation is not None:
                self.pre_aggregation.maybe_checkpoint(model, loss, metrics)
            else:
                log(INFO, "No Pre-aggregation checkpoint specified. Skipping.")
        elif mode == CheckpointMode.POST_AGGREGATION:
            if self.post_aggregation is not None:
                self.post_aggregation.maybe_checkpoint(model, loss, metrics)
            else:
                log(INFO, "No Post-aggregation checkpoint specified. Skipping.")
        else:
            raise ValueError(f"Unrecognized mode for checkpointing: {str(mode)}")
