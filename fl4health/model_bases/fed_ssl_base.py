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
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.pretrain = pretrain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.encoder(input)
        if self.pretrain:
            return self.projection_head(features)
        else:
            assert self.prediction_head is not None
            return self.prediction_head(features)
