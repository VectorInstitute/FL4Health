import torch.nn as nn
from torch import Tensor
from fl4health.utils.functions import bernoulli_sample
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Optional, Union
import copy

class MaskedLinear(nn.Linear):
    """
    Implementation of masked linear layers. 
    
    Like regular linear layers (i.e., nn.Linear module), a masked linear layer has a weight and a bias. 
    However, the weight and the bias do not receive gradient in back propagation. 
    Instead, two score tensors - one for the weight and another for the bias - are maintained.
    In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores, 
    which are then used to produce binary masks via bernoulli sampling. 
    Finally, the binary masks are  applied to the weight and the bias. During training,
    gradients with respect to the score tensors are computed and used to update the score tensors.
    
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

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype), requires_grad=False)
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_scores', None)
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
    """
    Implementation of masked Conv1d layers in a manner that is analogous to MaskedLinear.
    """
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
        padding_mode: str = 'zeros',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
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
            dtype
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias_scores', None)
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
            padding_mode=conv_module.padding_mode
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
        padding_mode: str = 'zeros',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
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
            dtype
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias_scores', None)
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
            padding_mode=conv_module.padding_mode
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
        padding_mode: str = 'zeros',
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
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
            dtype
        )
        self.weight.requires_grad = False
        self.weight_scores = Parameter(torch.randn_like(self.weight), requires_grad=True)
        if bias:
            assert self.bias is not None
            self.bias_scores = Parameter(torch.randn_like(self.bias), requires_grad=True)
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias_scores', None)
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
            padding_mode=conv_module.padding_mode
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


def convert_to_masked_model(original_model: nn.Module) -> nn.Module:
    masked_model = copy.deepcopy(original_model)
    for name, module in original_model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(masked_model, name, MaskedLinear.from_pretrained(module))
        elif isinstance(module, nn.Conv1d):
            setattr(masked_model, name, MaskedConv1d.from_pretrained(module))
        elif isinstance(module, nn.Conv2d):
            setattr(masked_model, name, MaskedConv2d.from_pretrained(module))
        elif isinstance(module, nn.Conv3d):
            setattr(masked_model, name, MaskedConv3d.from_pretrained(module))
    return masked_model



def is_masked_module(module: nn.Module) -> bool:
    return isinstance(module, (MaskedLinear, MaskedConv1d, MaskedConv2d, MaskedConv3d))
MaskedModule = Union[MaskedLinear, MaskedConv1d, MaskedConv2d, MaskedConv3d]