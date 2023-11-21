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

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        x = self.base_module.forward(input)
        features = self.projection_module.forward(x) if self.projection_module else x
        preds = self.head_module.forward(features)
        # Return preds and features as seperate dictionary as in fenda base
        return {"prediction": preds}, {"features": features.reshape(len(features), -1)}
