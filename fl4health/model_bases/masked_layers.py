import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.parameter import Parameter

from fl4health.utils.functions import bernoulli_sample


class MaskedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Implementation of masked linear layers.

        Like regular linear layers (i.e., nn.Linear module), a masked linear layer has a weight and a bias.
        However, the weight and the bias do not receive gradient in back propagation.
        Instead, two score tensors - one for the weight and another for the bias - are maintained.
        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores,
        which are then used to produce binary masks via bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training,
        gradients with respect to the score tensors are computed and used to update the score tensors.

        Note: the scores are not assumed to be bounded between 0 and 1.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``

        Attributes:
            weight: weights of the module.
            bias:  bias of the module.
            weight_score: learnable scores for the weights. Has the same shape as weight.
            bias_score: learnable scores for the bias. Has the same shape as bias.
        """
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
            self.register_parameter("bias", None)
            self.register_parameter("bias_scores", None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        # Produce probability scores and perform bernoulli sampling
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
    def from_pretrained(cls, linear_module: nn.Linear) -> "MaskedLinear":
        """
        Return an instance of MaskedLinear whose weight and bias have the same values as those of linear_module.
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


class MaskedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Implementation of masked Conv1d layers.

        Like regular Conv1d layers (i.e., nn.Conv1d module), a masked convolutional layer has a weight
        (i.e., convolutional filter) and a (optional) bias.
        However, the weight and the bias do not receive gradient in back propagation.
        Instead, two score tensors - one for the weight and another for the bias - are maintained.
        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores,
        which are then used to produce binary masks via bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training,
        gradients with respect to the score tensors are computed and used to update the score tensors.

        Note: the scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to both sides of
                the input. Default: 0
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``

        Attributes:
            weight: weights of the module.
            bias:  bias of the module.
            weight_score: learnable scores for the weights. Has the same shape as weight.
            bias_score: learnable scores for the bias. Has the same shape as bias.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias.requires_grad = False
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter("bias_scores", None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None
        return self._conv_forward(input, weight=masked_weight, bias=masked_bias)

    @classmethod
    def from_pretrained(cls, conv_module: nn.Conv1d) -> "MaskedConv1d":
        """
        Return an instance of MaskedConv1d whose weight and bias have the same values as those of conv_module.
        """
        has_bias = conv_module.bias is not None
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(conv_module.kernel_size)
        stride_ = _single(conv_module.stride)
        padding_ = conv_module.padding if isinstance(conv_module.padding, str) else _single(conv_module.padding)
        dilation_ = _single(conv_module.dilation)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=conv_module.groups,
            bias=has_bias,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Implementation of masked Conv2d layers.

        Like regular Conv2d layers (i.e., nn.Conv2d module), a masked convolutional layer has a weight
        (i.e., convolutional filter) and a (optional) bias.
        However, the weight and the bias do not receive gradient in back propagation.
        Instead, two score tensors - one for the weight and another for the bias - are maintained.
        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores,
        which are then used to produce binary masks via bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training,
        gradients with respect to the score tensors are computed and used to update the score tensors.

        Note: the scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all four sides of
                the input. Default: 0
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``

        Attributes:
            weight: weights of the module.
            bias:  bias of the module.
            weight_score: learnable scores for the weights. Has the same shape as weight.
            bias_score: learnable scores for the bias. Has the same shape as bias.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias.requires_grad = False
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter("bias_scores", None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None
        return self._conv_forward(input, weight=masked_weight, bias=masked_bias)

    @classmethod
    def from_pretrained(cls, conv_module: nn.Conv2d) -> "MaskedConv2d":
        """
        Return an instance of MaskedConv2d whose weight and bias have the same values as those of conv_module.
        """
        has_bias = conv_module.bias is not None
        kernel_size_ = _pair(conv_module.kernel_size)
        stride_ = _pair(conv_module.stride)
        padding_ = conv_module.padding if isinstance(conv_module.padding, str) else _pair(conv_module.padding)
        dilation_ = _pair(conv_module.dilation)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=conv_module.groups,
            bias=has_bias,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


class MaskedConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Implementation of masked Conv2d layers.

        Like regular Conv3d layers (i.e., nn.Conv3d module), a masked convolutional layer has a weight
        (i.e., convolutional filter) and a (optional) bias.
        However, the weight and the bias do not receive gradient in back propagation.
        Instead, two score tensors - one for the weight and another for the bias - are maintained.
        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores,
        which are then used to produce binary masks via bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training,
        gradients with respect to the score tensors are computed and used to update the score tensors.

        Note: the scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of
                the input. Default: 0
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

        Attributes:
            weight: weights of the module.
            bias:  bias of the module.
            weight_score: learnable scores for the weights. Has the same shape as weight.
            bias_score: learnable scores for the bias. Has the same shape as bias.
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias.requires_grad = False
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter("bias_scores", None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None
        return self._conv_forward(input, weight=masked_weight, bias=masked_bias)

    @classmethod
    def from_pretrained(cls, conv_module: nn.Conv3d) -> "MaskedConv3d":
        """
        Return an instance of MaskedConv3d whose weight and bias have the same values as those of conv_module.
        """
        has_bias = conv_module.bias is not None
        kernel_size_ = _triple(conv_module.kernel_size)
        stride_ = _triple(conv_module.stride)
        padding_ = conv_module.padding if isinstance(conv_module.padding, str) else _triple(conv_module.padding)
        dilation_ = _triple(conv_module.dilation)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=conv_module.groups,
            bias=has_bias,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


def convert_to_masked_model(original_model: nn.Module) -> nn.Module:
    """
    Given a model, convert every one of its linear or convolutional layer to a masked layer
    of the same kind, as defined in the classes above.
    """
    masked_model = copy.deepcopy(original_model)
    for name, module in original_model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, MaskedLinear):
            setattr(masked_model, name, MaskedLinear.from_pretrained(module))
        elif isinstance(module, nn.Conv1d) and not isinstance(module, MaskedConv1d):
            setattr(masked_model, name, MaskedConv1d.from_pretrained(module))
        elif isinstance(module, nn.Conv2d) and not isinstance(module, MaskedConv2d):
            setattr(masked_model, name, MaskedConv2d.from_pretrained(module))
        elif isinstance(module, nn.Conv3d) and not isinstance(module, MaskedConv3d):
            setattr(masked_model, name, MaskedConv3d.from_pretrained(module))
    return masked_model


def is_masked_module(module: nn.Module) -> bool:
    return isinstance(module, (MaskedLinear, MaskedConv1d, MaskedConv2d, MaskedConv3d))
