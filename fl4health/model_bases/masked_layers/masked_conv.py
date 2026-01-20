from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.parameter import Parameter

from fl4health.utils.functions import bernoulli_sample


class MaskedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``Conv1d`` layers.

        Like regular ``Conv1d`` layers (i.e., ``nn.Conv1d`` module), a masked convolutional layer has a weight (i.e.,
        convolutional filter) and a (optional) bias. However, the weight and the bias do not receive gradient in
        back propagation. Instead, two score tensors - one for the weight and another for the bias - are maintained.

        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores,
        which are then used to produce binary masks via Bernoulli sampling. Finally, the binary masks are applied to
        the weight and the bias. During training, gradients with respect to the score tensors are computed and used to
        update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int, tuple or str, optional): Padding added to both sides of the input. Default: 0.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: weights of the module.
        # bias:  bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward for the mask 1D Convolution.

        Args:
            input (Tensor): input tensor for the layer

        Returns:
            (Tensor): output tensor for the convolution
        """
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
    def from_pretrained(cls, conv_module: nn.Conv1d) -> MaskedConv1d:
        """
        Return an instance of ``MaskedConv1d`` whose weight and bias have the same values as those of ``conv_module``.

        Args:
            conv_module (nn.Conv1d): Module to be converted.

        Returns:
            (MaskedConv1d): Module with masked layers added to enable FedPM training.
        """
        has_bias = conv_module.bias is not None
        # we create new variables below to make mypy happy since kernel_size has
        # type int | tuple[int] and kernel_size_ has type tuple[int]
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
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``Conv2d`` layers.

        Like regular ``Conv2d`` layers (i.e., ``nn.Conv2d`` module), a masked convolutional layer has a weight (i.e.,
        convolutional filter) and a (optional) bias. However, the weight and the bias do not receive gradient in back
        propagation. Instead, two score tensors - one for the weight and another for the bias - are maintained.
        In the forward pass, the score tensors are transformed by the Sigmoid function into probability scores, which
        are then used to produce binary masks via Bernoulli sampling. Finally, the binary masks are applied to the
        weight and the bias. During training, gradients with respect to the score tensors are computed and used to
        update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: weights of the module.
        # bias:  bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward for the Masked 2D Convolution.

        Args:
            input (Tensor): input tensor for the layer.

        Returns:
            (Tensor): output tensor for the convolution
        """
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
    def from_pretrained(cls, conv_module: nn.Conv2d) -> MaskedConv2d:
        """
        Return an instance of ``MaskedConv2d`` whose weight and bias have the same values as those of ``conv_module``.

        Args:
            conv_module (nn.Conv2d): Module to be converted.

        Returns:
            (MaskedConv2d): Module with masked layers to enable FedPM.
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
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``Conv3d`` layers.

        Like regular ``Conv3d`` layers (i.e., ``nn.Conv3d`` module), a masked convolutional layer has a weight (i.e.,
        convolutional filter) and a (optional) bias. However, the weight and the bias do not receive gradient in back
        propagation. Instead, two score tensors - one for the weight and another for the bias - are maintained. In the
        forward pass, the score tensors are transformed by the Sigmoid function into probability scores, which are
        then used to produce binary masks via Bernoulli sampling. Finally, the binary masks are applied to the weight
        and the bias. During training, gradients with respect to the score tensors are computed and used to update the
        score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'zeros'``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight: weights of the module.
        # bias:  bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward for the Masked 3D Convolution.

        Args:
            input (Tensor): input tensor for the layer.

        Returns:
            (Tensor): output tensor for the convolution.
        """
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
    def from_pretrained(cls, conv_module: nn.Conv3d) -> MaskedConv3d:
        """
        Return an instance of ``MaskedConv3d`` whose weight and bias have the same values as those of ``conv_module``.

        Args:
            conv_module (nn.Conv3d): Module to convert.

        Returns:
            (MaskedConv3d): Module with mask layers added to enable FedPM.
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


class MaskedConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``ConvTranspose1d`` layers. For more information on transposed convolution,
        please see the PyTorch implementation of ``nn.Conv1d``.

        (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose1d)

        Like regular ``ConvTranspose1d`` layers (i.e., ``nn.ConvTranspose1d`` module), a masked transpose
        convolutional layer has a weight (i.e., convolutional filter) and a (optional) bias. However, the weight and
        the bias do not receive gradient in back propagation. Instead, two score tensors - one for the weight and
        another for the bias - are maintained. In the forward pass, the score tensors are transformed by the Sigmoid
        function into probability scores, which are then used to produce binary masks via Bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training, gradients with respect to
        the score tensors are computed and used to update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to
                both sides of the input. Default: 0.
            output_padding (int or tuple, optional): Additional size added to one side of the output shape. Default: 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            padding_mode (str, optional): Mode to be used in padding the input image for processing. Defaults to
                "zeros".
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight (Tensor): weights of the module.
        # bias (Tensor):   bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
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

    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor:
        """
        Forward for the ``MaskedConvTranspose1D``.

        Args:
            input (Tensor): input to be mapped with the module.
            output_size (list[int] | None, optional): Desired output from the transpose. Defaults to None.

        Raises:
            ValueError: If something other than "zeros" padding has been requested.

        Returns:
            (Tensor): Output tensors.
        """
        # Note: the same check is already present in super().__init__
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")
        assert isinstance(self.padding, tuple)

        # (The type ignore below is just used to resolve some small typing issue.)
        # One cannot replace List by Tuple or Sequence in "_output_padding"
        # because TorchScript does not support `Sequence[T]` or `tuple[T, ...]`.
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims=1,
            dilation=self.dilation,  # type: ignore[arg-type]
        )

        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None

        return F.conv_transpose1d(
            input, masked_weight, masked_bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )

    @classmethod
    def from_pretrained(cls, conv_module: nn.ConvTranspose1d) -> MaskedConvTranspose1d:
        """
        Return an instance of ``MaskedConvTranspose1d`` whose weight and bias have the same values as those of
        ``conv_module``.

        Args:
            conv_module (nn.ConvTranspose1d): Target module to be converted.

        Returns:
            (MaskedConvTranspose1d): Module with masked layers to enable FedPM.
        """
        has_bias = conv_module.bias is not None
        # we create new variables below to make mypy happy since kernel_size has
        # type int | tuple[int] and kernel_size_ has type tuple[int]
        kernel_size_ = _single(conv_module.kernel_size)
        stride_ = _single(conv_module.stride)
        padding_ = _single(conv_module.padding)
        dilation_ = _single(conv_module.dilation)
        output_padding_ = _single(conv_module.output_padding)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            output_padding=output_padding_,
            groups=conv_module.groups,
            bias=has_bias,
            dilation=dilation_,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


class MaskedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``ConvTranspose2d`` layers. For more information on transposed convolution,
        please see the PyTorch implementation of ``nn.Conv2d``.

        (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose2d)

        Like regular ``ConvTranspose2d`` layers (i.e., ``nn.ConvTranspose2d`` module), a masked transpose
        convolutional layer has a weight (i.e., convolutional filter) and a (optional) bias. However, the weight and
        the bias do not receive gradient in back propagation. Instead, two score tensors - one for the weight and
        another for the bias - are maintained. In the forward pass, the score tensors are transformed by the
        Sigmoid function into probability scores, which are then used to produce binary masks via Bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training, gradients with respect to
        the score tensors are computed and used to update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding will be added
                to both sides of each dimension in the input. Default: 0.
            output_padding (int or tuple, optional): Additional size added to one side of each dimension in the
                output shape. Default: 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            padding_mode (str, optional): Mode to be used in padding the input image for processing. Defaults to
                "zeros".
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight (Tensor): weights of the module.
        # bias (Tensor):   bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
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

    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor:
        """
        Maps input tensor through the ``MaskedConvTranspose2D`` module.

        Args:
            input (Tensor): tensor to be mapped.
            output_size (list[int] | None, optional): Desired output size from the module. Defaults to None.

        Raises:
            ValueError: Thrown if anything except "zeros" padding is requested.

        Returns:
            (Tensor): Mapped tensor.
        """
        # Note: the same check is already present in super().__init__
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")
        assert isinstance(self.padding, tuple)

        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims=2,
            dilation=self.dilation,  # type: ignore[arg-type]
        )

        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None

        return F.conv_transpose2d(
            input, masked_weight, masked_bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )

    @classmethod
    def from_pretrained(cls, conv_module: nn.ConvTranspose2d) -> MaskedConvTranspose2d:
        """
        Return an instance of ``MaskedConvTranspose2d`` whose weight and bias have the same values as those of
        ``conv_module``.

        Args:
            conv_module (nn.ConvTranspose2d): Target module to be converted.

        Returns:
            (MaskedConvTranspose2d): Module with mask layers added to enable FedPM.
        """
        has_bias = conv_module.bias is not None
        # we create new variables below to make mypy happy since kernel_size has
        # type int | tuple[int] and kernel_size_ has type tuple[int]
        kernel_size_ = _pair(conv_module.kernel_size)
        stride_ = _pair(conv_module.stride)
        padding_ = _pair(conv_module.padding)
        dilation_ = _pair(conv_module.dilation)
        output_padding_ = _pair(conv_module.output_padding)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            output_padding=output_padding_,
            groups=conv_module.groups,
            bias=has_bias,
            dilation=dilation_,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module


class MaskedConvTranspose3d(nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Implementation of masked ``ConvTranspose3d`` layers. For more information on transposed convolution,
        please see the PyTorch implementation of ``nn.Conv3d``.

        (https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#ConvTranspose3d)

        Like regular ``ConvTranspose3d`` layers (i.e., ``nn.ConvTranspose3d`` module), a masked transpose
        convolutional layer has a weight (i.e., convolutional filter) and a (optional) bias. However, the weight and
        the bias do not receive gradient in back propagation. Instead, two score tensors - one for the weight and
        another for the bias - are maintained. In the forward pass, the score tensors are transformed by the Sigmoid
        function into probability scores, which are then used to produce binary masks via Bernoulli sampling.
        Finally, the binary masks are applied to the weight and the bias. During training, gradients with respect to
        the score tensors are computed and used to update the score tensors.

        **NOTE**: The scores are not assumed to be bounded between 0 and 1.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.
            padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to
                both sides of each dimension in the input. Default: 0.
            output_padding (int or tuple, optional): Additional size added to one side of each dimension in the
                output shape. Default: 0.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
            padding_mode (str, optional): Mode to be used in padding the input image for processing. Defaults to
                "zeros".
            device (torch.device | None, optional): Device to which this module should be sent. Defaults to None.
            dtype (torch.dtype | None, optional): Type of the tensors. Defaults to None.
        """
        # Attributes:
        # weight (Tensor): weights of the module.
        # bias (Tensor):   bias of the module.
        # weight_score: learnable scores for the weights. Has the same shape as weight.
        # bias_score: learnable scores for the bias. Has the same shape as bias.
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
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

    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor:
        """
        Maps the input tensor with ``MaskedConvTranspose3D``.

        Args:
            input (Tensor): Tensor to be mapped.
            output_size (list[int] | None, optional): Desired output size from the transpose. Defaults to None.

        Raises:
            ValueError: Throws if anything except "zeros" padding is requested.

        Returns:
            (Tensor): Mapped tensor.
        """
        # Note: the same check is already present in super().__init__
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")
        assert isinstance(self.padding, tuple)

        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims=3,
            dilation=self.dilation,  # type: ignore[arg-type]
        )

        weight_prob_scores = torch.sigmoid(self.weight_scores)
        weight_mask = bernoulli_sample(weight_prob_scores)
        masked_weight = weight_mask * self.weight
        if self.bias is not None:
            bias_prob_scores = torch.sigmoid(self.bias_scores)
            bias_mask = bernoulli_sample(bias_prob_scores)
            masked_bias = bias_mask * self.bias
        else:
            masked_bias = None

        return F.conv_transpose3d(
            input, masked_weight, masked_bias, self.stride, self.padding, output_padding, self.groups, self.dilation
        )

    @classmethod
    def from_pretrained(cls, conv_module: nn.ConvTranspose3d) -> MaskedConvTranspose3d:
        """
        Return an instance of ``MaskedConvTranspose3d`` whose weight and bias have the same values as those of
        ``conv_module``.

        Args:
            conv_module (nn.ConvTranspose3d): Target module to be converted.

        Returns:
            (MaskedConvTranspose3d): Module with masked layers added to enable FedPM.
        """
        has_bias = conv_module.bias is not None
        # we create new variables below to make mypy happy since kernel_size has
        # type int | tuple[int] and kernel_size_ has type tuple[int]
        kernel_size_ = _triple(conv_module.kernel_size)
        stride_ = _triple(conv_module.stride)
        padding_ = _triple(conv_module.padding)
        dilation_ = _triple(conv_module.dilation)
        output_padding_ = _triple(conv_module.output_padding)
        masked_conv_module = cls(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            output_padding=output_padding_,
            groups=conv_module.groups,
            bias=has_bias,
            dilation=dilation_,
            padding_mode=conv_module.padding_mode,
        )
        masked_conv_module.weight = Parameter(conv_module.weight.clone().detach(), requires_grad=False)
        masked_conv_module.weight_scores = Parameter(torch.randn_like(masked_conv_module.weight), requires_grad=True)
        if has_bias:
            assert conv_module.bias is not None
            masked_conv_module.bias = Parameter(conv_module.bias.clone().detach(), requires_grad=False)
            masked_conv_module.bias_scores = Parameter(torch.randn_like(masked_conv_module.bias), requires_grad=True)
        return masked_conv_module
