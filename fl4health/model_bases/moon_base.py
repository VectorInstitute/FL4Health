from typing import Dict, Optional

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

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # input is expected to be of shape (batch_size, *)
        x = self.base_module.forward(input)
        if self.projection_module:
            p = self.projection_module.forward(x)
        else:
            p = x
        output = self.head_module.forward(p)
        return {"prediction": output, "features": p.reshape(len(p), -1)}
