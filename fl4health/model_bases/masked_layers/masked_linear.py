from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from fl4health.utils.functions import bernoulli_sample


class MaskedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked linear layers.

        Like regular linear layers (i.e., ``nn.Linear module``), a masked linear layer has a weight and a bias.
        However, the weight and the bias do not receive gradient in back propagation. Instead, two score tensors - one
        for the weight and another for the bias - are maintained. In the forward pass, the score tensors are
        transformed by the Sigmoid function into probability scores, which are then used to produce binary masks via
        Bernoulli sampling. Finally, the binary masks are applied to the weight and the bias. During training,
        gradients with respect to the score tensors are computed and used to update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
            bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: weights of the module.
        # bias:  bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
        super().__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias.requires_grad = False
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter("bias_scores", None)

    def forward(self, input: Tensor) -> Tensor:
        """
        Mapping function for the ``MaskedLinear`` layer.

        Args:
            input (Tensor): input tensor to be transformed.

        Returns:
            (Tensor): output tensor from the layer.
        """
        # Produce probability scores and perform Bernoulli sampling
        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight

        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None

        # Apply the masks to weight and bias
        return F.linear(input, masked_weight, masked_bias)

    @classmethod
    def from_pretrained(cls, linear_module: nn.Linear) -> MaskedLinear:
        """
        Return an instance of ``MaskedLinear`` whose weight and bias have the same values as those of
        ``linear_module``.

        Args:
            linear_module (nn.Linear): Target layer to be transformed.

        Returns:
            (MaskedLinear): New copy of the provided module with masked layers inserted to enable FedPM.
        """
        has_bias = linear_module.bias is not None
        masked_linear_module = cls(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=has_bias,
        )
        masked_linear_module.weight = Parameter(linear_module.weight.clone().detach(), requires_grad=False)
        masked_linear_module.weight_scores = Parameter(torch.randn_like(linear_module.weight), requires_grad=True)
        if has_bias:
            masked_linear_module.bias = Parameter(linear_module.bias.clone().detach(), requires_grad=False)
            masked_linear_module.bias_scores = Parameter(torch.randn_like(linear_module.bias), requires_grad=True)
        return masked_linear_module
