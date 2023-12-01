import copy
from typing import Dict, List

import torch
import torch.nn as nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class ApflModule(PartialLayerExchangeModel):
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

    def global_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.global_model(input)

    def local_forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.local_model(input)

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Forward return dictionary because APFL has multiple different prediction types
        global_logits = self.global_forward(input)
        local_logits = self.local_forward(input)
        personal_logits = self.alpha * local_logits + (1.0 - self.alpha) * global_logits
        preds = {"personal": personal_logits, "global": global_logits, "local": local_logits}
        return preds

    def update_alpha(self) -> None:
        # Updates to mixture parameter follow original implementation
        # https://github.com/MLOPTPSU/FedTorch/blob
        # /ab8068dbc96804a5c1a8b898fd115175cfebfe75/fedtorch/comms/utils/flow_utils.py#L240

        # Need to filter out frozen parameters, as they have no grad object
        local_parameters = [
            local_params for local_params in self.local_model.parameters() if local_params.requires_grad
        ]
        global_parameters = [
            global_params for global_params in self.global_model.parameters() if global_params.requires_grad
        ]

        # Accumulate gradient of alpha across layers
        grad_alpha: float = 0.0
        for local_p, global_p in zip(local_parameters, global_parameters):
            local_grad = local_p.grad
            global_grad = global_p.grad
            assert local_grad is not None and global_grad is not None
            dif = local_p - global_p
            grad = torch.tensor(self.alpha) * local_grad + torch.tensor(1.0 - self.alpha) * global_grad
            grad_alpha += torch.mul(dif, grad).sum().detach().cpu().numpy().item()

        # This update constant of 0.02 is not referenced in the paper
        # but is present in the official implementation and other ones I have seen
        # Not sure its function, just adding a number proportional to alpha to the grad
        # Leaving in for consistency with official implementation
        grad_alpha += 0.02 * self.alpha
        alpha = self.alpha - self.alpha_lr * grad_alpha
        # Clip alpha to be between [0, 1]
        alpha = max(min(alpha, 1), 0)
        self.alpha = alpha

    def layers_to_exchange(self) -> List[str]:
        layers_to_exchange: List[str] = [
            layer for layer in self.state_dict().keys() if layer.startswith("global_model.")
        ]
        return layers_to_exchange
