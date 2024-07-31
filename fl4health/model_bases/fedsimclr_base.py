from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class FedSimClrModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module = nn.Identity(),
        prediction_head: Optional[nn.Module] = None,
        pretrain: bool = True,
    ) -> None:
        """
        Model base to train SimCLR (https://arxiv.org/pdf/2002.05709)
        in a federated manner presented in (https://arxiv.org/pdf/2207.09158).
        Can be used in pretraining and optionally finetuning.

        Args:
            encoder (nn.Module): Encoder that extracts a feature vector.
                given an input sample.
            projection_head (nn.Module): Projection Head that maps output
                of encoder to final representation used in contrastive loss
                for pretraining stage. Defaults to identity transformation.
            prediction_head (Optional[nn.Module]): Prediction head that maps
                output of encoder to prediction in the finetuning stage.
                Defaults to None.
            pretrain (bool): Determines whether or not to use the projection_head
                (True) or the prediction_head (False). Defaults to True.
        """
        super().__init__()

        assert not (
            prediction_head is None and not pretrain
        ), "Model with pretrain==False must have prediction head (ie not None)"

        self.encoder = encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.pretrain = pretrain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input)
        if self.pretrain:
            return self.projection_head(features)
        else:
            assert (
                self.prediction_head is not None
            ), "Model with pretrain==False must have prediction_head (ie not None)"
            return self.prediction_head(features)

    @staticmethod
    def load_pretrained_model(model_path: Path) -> FedSimClrModel:
        prev_model = torch.load(model_path)
        ssl_model = FedSimClrModel(
            encoder=prev_model.encoder,
            projection_head=prev_model.projection_head,
            prediction_head=prev_model.prediction_head,
            pretrain=False,
        )
        return ssl_model
