from typing import List

import torch
import torch.nn as nn


class WeightDriftLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

    def _compute_weight_difference_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.linalg.norm(x - y), 2.0)

    def forward(self, target_model: nn.Module, constraint_tensors: List[torch.Tensor], weight: float) -> torch.Tensor:
        # move model and tensors to device if needed
        target_model = target_model.to(self.device)
        constraint_tensors = [constraint_tensor.to(self.device) for constraint_tensor in constraint_tensors]

        model_weights = [layer_weights for layer_weights in target_model.parameters()]
        assert len(constraint_tensors) == len(model_weights)
        assert len(model_weights) > 0

        layer_inner_products: List[torch.Tensor] = [
            self._compute_weight_difference_inner_product(constraint_layer_weights, model_layer_weights)
            for constraint_layer_weights, model_layer_weights in zip(constraint_tensors, model_weights)
        ]

        # Network l2 inner product tensor
        # NOTE: Scaling by 1/2 is for grad consistency.
        return (weight / 2.0) * torch.stack(layer_inner_products).sum()
