import torch
from torch import nn


class WeightDriftLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
    ) -> None:
        r"""
        Used to compute the \(l_2\)-inner product between a Torch model and a reference set of weights
        corresponding to a past version of that model.

        Args:
            device (torch.device): Device on which the loss should be computed.
        """
        super().__init__()
        self.device = device

    def _compute_weight_difference_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the \\(l_2\\)-inner product between two tensors. This amounts to the Frobenius norm of the
        difference between the tensors \\(\\Vert x - y \\Vert_F\\).

        Args:
            x (torch.Tensor): first tensor
            y (torch.Tensor): second tensor

        Returns:
            (torch.Tensor): Frobenius norm of their difference
        """
        return torch.pow(torch.linalg.norm(x - y), 2.0)

    def forward(self, target_model: nn.Module, constraint_tensors: list[torch.Tensor], weight: float) -> torch.Tensor:
        r"""
        Compute the \(l_2\)-inner product between a Torch model and a reference set of weights in a differentiable
        way. The ```constraint_tensors``` are frozen.

        Args:
            target_model (nn.Module): Model being constrained by the ``constraint_tensors``. Weights are
                differentiable.
            constraint_tensors (list[torch.Tensor]): Tensors corresponding to a previous version of the
                ``target_model``.
            weight (float): Weight with which to scale the loss.

        Returns:
            (torch.Tensor): Loss value.
        """
        # move model and tensors to device if needed
        target_model = target_model.to(self.device)
        constraint_tensors = [constraint_tensor.to(self.device) for constraint_tensor in constraint_tensors]

        model_weights = list(target_model.parameters())
        assert len(constraint_tensors) == len(model_weights)
        assert len(model_weights) > 0

        layer_inner_products: list[torch.Tensor] = [
            self._compute_weight_difference_inner_product(constraint_layer_weights, model_layer_weights)
            for constraint_layer_weights, model_layer_weights in zip(constraint_tensors, model_weights)
        ]

        # Network l2 inner product tensor
        # NOTE: Scaling by 1/2 is for grad consistency.
        return (weight / 2.0) * torch.stack(layer_inner_products).sum()
