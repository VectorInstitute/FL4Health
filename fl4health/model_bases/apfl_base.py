import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class APFLModule(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        adaptive_alpha: bool = True,
        alpha: float = 0.5,
        alpha_lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.local_model: nn.Module = model
        self.global_model: nn.Module = copy.deepcopy(model)

        self.adaptive_alpha = adaptive_alpha
        self.alpha = alpha
        self.alpha_lr = alpha_lr

    def forward(self, input: torch.Tensor, personal: bool) -> Dict[str, torch.Tensor]:

        if not personal:
            global_logits = self.global_model(input)
            return {"global": global_logits}

        global_logits = self.global_model(input)
        local_logits = self.local_model(input)
        personal_logits = self.alpha * local_logits + (1.0 - self.alpha) * global_logits
        results = {"personal": personal_logits, "local": local_logits}

        return results

    def update_alpha(self) -> None:
        grad_alpha: float = 0.0
        for local_p, global_p in zip(self.local_model.parameters(), self.global_model.parameters()):
            dif = local_p - global_p
            grad = self.alpha * local_p.grad + (1.0 - self.alpha) * global_p.grad
            grad_alpha += dif.flatten().dot(grad.flatten()).detach().numpy()

        grad_alpha += 0.02 * self.alpha
        alpha = self.alpha - self.alpha_lr * grad_alpha
        alpha = np.clip(alpha, 0, 1)
        self.alpha = alpha

    def layers_to_exchange(self) -> List[str]:
        layers_to_exchange: List[str] = [layer for layer in self.state_dict().keys() if "global_model" in layer]
        return layers_to_exchange
