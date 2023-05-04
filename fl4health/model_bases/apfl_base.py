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
        # Updates to mixture parameter follow original implementation
        # https://github.com/MLOPTPSU/FedTorch/blob
        # /ab8068dbc96804a5c1a8b898fd115175cfebfe75/fedtorch/comms/utils/flow_utils.py#L240

        # Accumulate gradient of alpha across layers
        grad_alpha: float = 0.0
        for local_p, global_p in zip(self.local_model.parameters(), self.global_model.parameters()):
            local_grad = local_p.grad
            global_grad = global_p.grad
            assert local_grad is not None and global_grad is not None
            dif = local_p - global_p
            grad = torch.tensor(self.alpha) * local_grad + torch.tensor(1.0 - self.alpha) * global_grad
            grad_alpha += torch.mul(dif, grad).sum().detach().numpy()

        # This update constant of 0.02 is not referenced in the paper
        # but is present in the official implementation and other ones I have seen
        # Not sure its function, just adding a number proportional to alpha to the grad
        # Leaving in for consistency with official implementation
        grad_alpha += 0.02 * self.alpha
        alpha = self.alpha - self.alpha_lr * grad_alpha
        alpha = np.clip(alpha, 0, 1)
        self.alpha = alpha

    def layers_to_exchange(self) -> List[str]:
        layers_to_exchange: List[str] = [layer for layer in self.state_dict().keys() if "global_model" in layer]
        return layers_to_exchange
