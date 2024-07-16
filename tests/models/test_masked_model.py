import torch.nn as nn

from fl4health.model_bases.masked_model import (
    MaskedConv1d,
    MaskedConv2d,
    MaskedConv3d,
    MaskedLinear,
    convert_to_masked_model,
)
from tests.test_utils.models_for_test import CompositeConvNet


def test_masked_linear_init() -> None:
    masked_linear_module = MaskedLinear(in_features=8, out_features=16, bias=True)
    # check that the requires_grad property is properly set.
    assert not masked_linear_module.weight.requires_grad
    assert masked_linear_module.weight_scores.requires_grad
    assert not masked_linear_module.bias.requires_grad
    assert masked_linear_module.bias_scores.requires_grad


def test_masked_linear_from_pretrained() -> None:
    linear_module = nn.Linear(in_features=8, out_features=16, bias=True)
    masked_linear_module = MaskedLinear.from_pretrained(linear_module=linear_module)

    assert not masked_linear_module.weight.requires_grad
    assert masked_linear_module.weight_scores.requires_grad
    assert masked_linear_module.bias is not None and not masked_linear_module.bias.requires_grad
    assert masked_linear_module.bias_scores.requires_grad

    assert (masked_linear_module.weight == linear_module.weight).all()
    assert (masked_linear_module.bias == linear_module.bias).all()


def test_masked_conv1d() -> None:
    masked_conv_module = MaskedConv1d(16, 33, 3, stride=2, bias=True)
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv1d_from_pretrained() -> None:
    conv_module = nn.Conv1d(16, 33, 3, stride=2, bias=True)
    masked_conv_module = MaskedConv1d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert (masked_conv_module.weight == conv_module.weight).all()
    assert (masked_conv_module.bias == conv_module.bias).all()


def test_masked_conv2d() -> None:
    masked_conv_module = MaskedConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), bias=True)
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv2d_from_pretrained() -> None:
    conv_module = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), bias=True)
    masked_conv_module = MaskedConv2d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert (masked_conv_module.weight == conv_module.weight).all()
    assert (masked_conv_module.bias == conv_module.bias).all()


def test_masked_conv3d() -> None:
    masked_conv_module = MaskedConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0), bias=True)
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv3d_from_pretrained() -> None:
    conv_module = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0), bias=True)
    masked_conv_module = MaskedConv3d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert (masked_conv_module.weight == conv_module.weight).all()
    assert (masked_conv_module.bias == conv_module.bias).all()


def test_convert_to_masked_model() -> None:
    model = CompositeConvNet()
    masked_model = convert_to_masked_model(original_model=model)
    assert isinstance(masked_model.conv1d, MaskedConv1d)
    assert isinstance(masked_model.conv2d, MaskedConv2d)
    assert isinstance(masked_model.conv3d, MaskedConv3d)
    assert isinstance(masked_model.linear, MaskedLinear)
