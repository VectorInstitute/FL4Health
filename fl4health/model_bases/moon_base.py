from typing import Dict, Optional

import torch
import torch.nn as nn

from fl4health.model_bases.warm_up_base import WarmUpModel


class MoonModel(WarmUpModel):
    def __init__(
        self,
        base_module: nn.Module,
        head_module: nn.Module,
        projection_module: Optional[nn.Module] = None,
        warm_up: bool = False,
        warmed_up_dir: Optional[str] = None,
    ) -> None:
        super().__init__(warm_up, warmed_up_dir)
        self.base_module = base_module
        self.projection_module = projection_module
        self.head_module = head_module

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.base_module.forward(input)
        if self.projection_module:
            p = self.projection_module.forward(x)
        else:
            p = x
        output = self.head_module.forward(p)
        return {"prediction": output, "features": p.view(len(p), -1)}
