from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MoonModel(nn.Module):
    def __init__(
        self, base_module: nn.Module, head_module: nn.Module, projection_module: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.projection_module = projection_module
        self.head_module = head_module
        # Features are forced to be flattened in this model, as it is expected to always be used with the contrastive
        # loss function. However, inheriting models, such as FedPer may override this variable.
        self.flatten_features = True

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        x = self.base_module.forward(input)
        features = self.projection_module.forward(x) if self.projection_module else x
        preds = {"prediction": self.head_module.forward(features)}
        features = (
            {"features": features} if not self.flatten_features else {"features": features.reshape(len(features), -1)}
        )
        # Return preds and features as seperate dictionary as in fenda base
        return preds, features
