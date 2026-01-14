from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter

from fl4health.utils.functions import bernoulli_sample


TorchShape = int | list[int] | torch.Size

BATCH_NORM_3D_INPUT_LENGTH = 5
BATCH_NORM_2D_INPUT_LENGTH = 4
BATCH_NORM_1D_INPUT_LENGTHS = {2, 3}


class MaskedLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: TorchShape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of the masked Layer Normalization module. When ``elementwise_affine`` is True, ``nn.LayerNorm``
        has a learnable weight and (optional) bias. For ``MaskedLayerNorm``, the weight and bias do not receive
        gradient in back propagation. Instead, two score tensors - one for the weight and another for the bias - are
        maintained. In the forward pass, the score tensors are transformed by the Sigmoid function into probability
        scores, which are then used to produce binary masks via Bernoulli sampling. Finally, the binary masks are
        applied to the weight and the bias. During training, gradients with respect to the score tensors are computed
        and used to update the score tensors.

        When ``elementwise_affine`` is False, ``nn.LayerNorm`` does not have weight or bias. Under this condition, both
        score tensors are None and ``MaskedLayerNorm`` acts in the same way as ``nn.LayerNorm``.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            normalized_shape (TorchShape): Input shape from an expected input. If a single integer is used, it is
                treated as a singleton list, and this module will normalize over the last dimension which is expected
                to be of that specific size.
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module has learnable per-element
                affine parameters initialized to ones (for weights) and zeros (for biases). Default: ``True``.
            bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
                ``elementwise_affine`` is ``True``). Default: ``True``.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: the weights of the module. The values are initialized to 1.
        # bias:   the bias of the module. The values are initialized to 0.
        # weight_score: learnable scores for the weights. Has the same shape as weight. When applied
        # to the default initial values of self.weight (i.e., all ones), this is equivalent to
        # randomly dropping out certain features.
        # bias_score: learnable scores for the bias. Has the same shape as bias. When applied to
        # the default initial values of self.bias (i.e., all zeros), it does not have any actual
        # effect. Thus, bias_score only influences training when MaskedLayerNorm is created
        # from some pretrained nn.LayerNorm module whose bias is not all zeros.
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if self.elementwise_affine:
            assert self.weight is not None
            self.weight.requires_grad = False
            self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
            if self.bias is not None:
                self.bias.requires_grad = False
                self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
            else:
                self.register_parameter("bias_scores", None)
        else:
            self.register_parameter("weight_scores", None)
            self.register_parameter("bias_scores", None)

    def forward(self, input: Tensor) -> Tensor:
        """
        Mapping function for the ``MaskedLayerNorm``.

        Args:
            input (Tensor): Tensor to be mapped by the layer.

        Returns:
            (Tensor): Output tensor after mapping of the input tensor.
        """
        if not self.elementwise_affine:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        assert self.weight is not None
        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None
        return F.layer_norm(input, self.normalized_shape, masked_weight, masked_bias, self.eps)

    @classmethod
    def from_pretrained(cls, layer_norm_module: nn.LayerNorm) -> MaskedLayerNorm:
        """
        Return an instance of ``MaskedLayerNorm`` whose weight and bias have the same values as those of
        ``layer_norm_module``.

        Args:
            layer_norm_module (nn.LayerNorm): Target module to be converted

        Returns:
            (MaskedLayerNorm): New copy of the provided module with mask layers added to enable FedPM
        """
        masked_layer_norm_module = cls(
            # layer_norm_module.normalized_shape is a tuple so we
            # simply transform it into torch.Size so it is compatible with
            # the constructor's type signature.
            normalized_shape=torch.Size(layer_norm_module.normalized_shape),
            eps=layer_norm_module.eps,
            elementwise_affine=layer_norm_module.elementwise_affine,
            bias=(layer_norm_module.bias is not None),
        )

        if layer_norm_module.elementwise_affine:
            assert layer_norm_module.weight is not None
            masked_layer_norm_module.weight = Parameter(layer_norm_module.weight.clone().detach(), requires_grad=False)
            masked_layer_norm_module.weight_scores = Parameter(
                torch.randn_like(layer_norm_module.weight), requires_grad=True
            )
            if layer_norm_module.bias is not None:
                masked_layer_norm_module.bias = Parameter(layer_norm_module.bias.clone().detach(), requires_grad=False)
                masked_layer_norm_module.bias_scores = Parameter(
                    torch.randn_like(layer_norm_module.bias), requires_grad=True
                )

        return masked_layer_norm_module


class _MaskedBatchNorm(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        r"""
        Base class for masked batch normalization modules of various dimensions. When affine is True,
        ``_BatchNorm`` has a learnable weight and bias. For ``_MaskedBatchNorm``, the weight and bias do not
        receive gradient in back propagation. Instead, two score tensors - one for the weight and another for the
        bias - are maintained. In the forward pass, the score tensors are transformed by the Sigmoid function
        into probability scores, which are then used to produce binary masks via Bernoulli sampling. Finally, the
        binary masks are applied to the weight and the bias. During training, gradients with respect to the score
        tensors are computed and used to update the score tensors.

        When affine is False, _BatchNorm does not have weight or bias. Under this condition, both score tensors
        are None and ``_MaskedBatchNorm`` acts in the same way as ``_BatchNorm``.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            num_features (int): Number of features or channels \(C\) of the input
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
            momentum (float | None, optional): The value used for the running_mean and ``running_var`` computation.
                Can be set to ``None`` for cumulative moving average (i.e. simple average). Defaults to 0.1.
            affine (bool, optional): A boolean value that when set to ``True``, this module has learnable affine
                parameters. Defaults to True.
            track_running_stats (bool, optional): A boolean value that when set to ``True``, this module tracks the
                running mean and variance, and when set to ``False``, this module does not track such statistics, and
                initializes statistics buffers :attr:`running_mean` and :attr:`running_var` as ``None``. When these
                buffers are ``None``, this module always uses batch statistics. in both training and eval modes.
                Defaults to True.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: the weights of the module. The values are initialized to 1.
        # bias:   the bias of the module. The values are initialized to 0.
        # weight_score: learnable scores for the weights. Has the same shape as weight. When applied
        # to the default initial values of self.weight (i.e., all ones), this is equivalent to
        # randomly dropping out certain features.
        # bias_score: learnable scores for the bias. Has the same shape as bias. When applied to
        # the default initial values of self.bias (i.e., all zeros), it does not have any actual
        # effect. Thus, bias_score only influences training when MaskedLayerNorm is created
        # from some pretrained nn.LayerNorm module whose bias is not all zeros.
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device=device, dtype=dtype)
        if self.affine:
            assert (self.weight is not None) and (self.bias is not None)
            self.weight.requires_grad = False
            self.bias.requires_grad = False
            self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter("weight_scores", None)
            self.register_parameter("bias_scores", None)

    def forward(self, input: Tensor) -> Tensor:
        """
        Mapping function for the ``_MaskedBatchNorm`` module.

        Args:
            input (Tensor): Tensor to be mapped via the ``_MaskedBatchNorm``

        Returns:
            (Tensor): Output tensor after mapping
        """
        self._check_input_dim(input)
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        if self.training and self.track_running_stats and self.num_batches_tracked is not None:  # type: ignore[has-type]
            self.num_batches_tracked.add_(1)  # type: ignore[has-type]
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        bn_training = True if self.training else self.running_mean is None and self.running_var is None

        # Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        # passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        # used for normalization (i.e. in eval mode when buffers are not None).
        if self.affine:
            assert (self.weight is not None) and (self.bias is not None)
            weight_prob_scores = torch.sigmoid(self.weight_scores)
            weight_mask = bernoulli_sample(weight_prob_scores)
            masked_weight = weight_mask * self.weight

            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_weight = None
            masked_bias = None

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            masked_weight,
            masked_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    @classmethod
    def from_pretrained(cls, batch_norm_module: _BatchNorm) -> _MaskedBatchNorm:
        """
        Mapping a ``_BatchNorm`` module to a ``_MaskedBatchNorm`` by injecting masked layers.

        Args:
            batch_norm_module (_BatchNorm): Module to be transformed to a masked module through layer insertion

        Returns:
            (_MaskedBatchNorm): New copy of the input module with masked layers to enable FedPM
        """
        masked_batch_norm_module = cls(
            num_features=batch_norm_module.num_features,
            eps=batch_norm_module.eps,
            momentum=batch_norm_module.momentum,
            affine=batch_norm_module.affine,
            track_running_stats=batch_norm_module.track_running_stats,
        )
        if batch_norm_module.affine:
            assert (batch_norm_module.weight is not None) and (batch_norm_module.bias is not None)
            masked_batch_norm_module.weight = Parameter(batch_norm_module.weight.clone().detach(), requires_grad=False)
            masked_batch_norm_module.weight_scores = Parameter(
                torch.randn_like(batch_norm_module.weight), requires_grad=True
            )
            masked_batch_norm_module.bias = Parameter(batch_norm_module.bias.clone().detach(), requires_grad=False)
            masked_batch_norm_module.bias_scores = Parameter(
                torch.randn_like(batch_norm_module.weight), requires_grad=True
            )
        return masked_batch_norm_module


class MaskedBatchNorm1d(_MaskedBatchNorm):
    """
    Applies (masked) Batch Normalization over a 2D or 3D input. Input shape should be ``(N, C)`` or ``(N, C, L)``,
    where ``N`` is the batch size, ``C`` is the number of features/channels, and ``L`` is the sequence length.
    """

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() not in BATCH_NORM_1D_INPUT_LENGTHS:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class MaskedBatchNorm2d(_MaskedBatchNorm):
    """
    Applies (masked) Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel
    dimension).
    """

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != BATCH_NORM_2D_INPUT_LENGTH:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class MaskedBatchNorm3d(_MaskedBatchNorm):
    """
    Applies (masked) Batch Normalization over a 5D input (a mini-batch of 3D inputs with additional channel
    dimension).
    """

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != BATCH_NORM_3D_INPUT_LENGTH:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
