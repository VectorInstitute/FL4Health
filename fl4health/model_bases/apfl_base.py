import copy

import torch
from torch import nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class ApflModule(PartialLayerExchangeModel):
    def __init__(
        self,
        model: nn.Module,
        adaptive_alpha: bool = True,
        alpha: float = 0.5,
        alpha_lr: float = 0.01,
    ) -> None:
        """
        Defines a model compatible with the APFL approach.

        Args:
            model (nn.Module): The underlying model architecture to be optimized. A twin of this model will be created
                to initialize a local and global version of this architecture.
            adaptive_alpha (bool, optional): Whether or not the mixing parameter \\(\\alpha\\) will be adapted
                during training. Predictions of the local and global models are combined using \\(\\alpha\\) to
                provide a final prediction. Defaults to True.
            alpha (float, optional): The initial value for the mixing parameter \\(\\alpha\\). Defaults to 0.5.
            alpha_lr (float, optional): The learning rate to be applied when adaptive \\(\\alpha\\) during training.
                If ``adaptive_alpha`` is False, then this parameter does nothing. Defaults to 0.01.
        """
        super().__init__()
        self.local_model: nn.Module = model
        self.global_model: nn.Module = copy.deepcopy(model)

        self.adaptive_alpha = adaptive_alpha
        self.alpha = alpha
        self.alpha_lr = alpha_lr

    def global_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward function that runs the input tensor through the **GLOBAL** model only.

        Args:
            input (torch.Tensor): tensor to be run through the global model

        Returns:
            (torch.Tensor): output from the global model only.
        """
        return self.global_model(input)

    def local_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward function that runs the input tensor through the **LOCAL** model only.

        Args:
            input (torch.Tensor): tensor to be run through the local model.

        Returns:
            (torch.Tensor): output from the local model only.
        """
        return self.local_model(input)

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward function for the full APFL model. This includes mixing of the global and local model predictions using
        \\(\\alpha\\). The predictions are combined as follows.

        \\[\\alpha \\cdot \\text{local_logits} + (1.0 - \\alpha) \\cdot \\text{global_logits}\\]

        Args:
            input (torch.Tensor): Input tensor to be run through both the local and global models

        Returns:
            (dict[str, torch.Tensor]): Final prediction after mixing predictions produced by the local and global
                models. This dictionary stores these predictions under the key "personal" while the local and global
                model predictions are stored under the keys "global" and "local."
        """
        # Forward return dictionary because APFL has multiple different prediction types
        global_logits = self.global_forward(input)
        local_logits = self.local_forward(input)
        personal_logits = self.alpha * local_logits + (1.0 - self.alpha) * global_logits
        return {"personal": personal_logits, "global": global_logits, "local": local_logits}

    def update_alpha(self) -> None:
        """
        Updates to mixture parameter follow original implementation:

        https://github.com/MLOPTPSU/FedTorch/blob/ab8068dbc96804a5c1a8b898fd115175cfebfe75/fedtorch/comms/utils/flow_utils.py#L240
        """  # noqa

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

    def layers_to_exchange(self) -> list[str]:
        """
        Specifies the model layers to be exchanged with the server. These are a fixed set of layers exchanged every
        round. For APFL, these are any layers associated with the ``global_model``. That is, none of the parameters
        of the local model are aggregated on the server side, nor is \\(\\alpha\\).

        Returns:
            (list[str]): Names of layers associated with the global model. These correspond to the layer names in the
                state dictionary of this entire module.
        """
        layers_to_exchange: list[str] = [layer for layer in self.state_dict() if layer.startswith("global_model.")]
        return layers_to_exchange
