from typing import Optional, Tuple

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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input is expected to be of shape (batch_size, *)
        x = self.base_module.forward(input)
        p = self.projection_module.forward(x) if self.projection_module else x
        output = self.head_module.forward(p)
        return output, p.view(len(p), -1)
