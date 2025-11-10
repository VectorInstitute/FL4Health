import torch
from torch import nn
from torch.nn.parameter import Parameter

from fl4health.model_bases.masked_layers.masked_conv import (
    MaskedConv1d,
    MaskedConv2d,
    MaskedConv3d,
    MaskedConvTranspose1d,
    MaskedConvTranspose2d,
    MaskedConvTranspose3d,
)
from fl4health.model_bases.masked_layers.masked_layers_utils import convert_to_masked_model
from fl4health.model_bases.masked_layers.masked_linear import MaskedLinear
from fl4health.model_bases.masked_layers.masked_normalization_layers import (
    MaskedBatchNorm1d,
    MaskedBatchNorm2d,
    MaskedBatchNorm3d,
    MaskedLayerNorm,
)
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds
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

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


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

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


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

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


def test_masked_conv_transposed_1d() -> None:
    masked_conv_module = MaskedConvTranspose1d(
        in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=True
    )
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv_transposed_1d_from_pretrained() -> None:
    conv_module = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=True)
    masked_conv_module = MaskedConvTranspose1d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


def test_masked_conv_transposed_2d() -> None:
    masked_conv_module = MaskedConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), bias=True)
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv_transposed_2d_from_pretrained() -> None:
    conv_module = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), bias=True)
    masked_conv_module = MaskedConvTranspose2d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


def test_masked_conv_transposed_3d() -> None:
    masked_conv_module = MaskedConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2), bias=True)
    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad


def test_masked_conv_transposed_3d_from_pretrained() -> None:
    conv_module = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2), bias=True)
    masked_conv_module = MaskedConvTranspose3d.from_pretrained(conv_module=conv_module)

    assert not masked_conv_module.weight.requires_grad
    assert masked_conv_module.weight_scores.requires_grad
    assert masked_conv_module.bias is not None and not masked_conv_module.bias.requires_grad
    assert masked_conv_module.bias_scores.requires_grad

    assert torch.allclose(masked_conv_module.weight, conv_module.weight)
    assert conv_module.bias is not None
    assert torch.allclose(masked_conv_module.bias, conv_module.bias)


def test_masked_layer_norm() -> None:
    set_all_random_seeds(42)
    masked_layer_norm_module = MaskedLayerNorm(10, elementwise_affine=True, bias=True)
    assert (masked_layer_norm_module.bias is not None) and (masked_layer_norm_module.weight is not None)
    assert (not masked_layer_norm_module.weight.requires_grad) and masked_layer_norm_module.weight_scores.requires_grad
    assert (not masked_layer_norm_module.bias.requires_grad) and masked_layer_norm_module.bias_scores.requires_grad

    # Test the forward pass
    input = torch.randn(10)
    output = masked_layer_norm_module(input)

    # output should be equivalent to a standard layer norm with the right weights dropped
    layer_norm_module = nn.LayerNorm(10)
    layer_norm_module.weight = Parameter(
        torch.Tensor([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]), requires_grad=True
    )
    output_target = layer_norm_module(input)

    assert torch.allclose(output, output_target, atol=1e-5)

    non_affine_layer_norm = nn.LayerNorm(10, elementwise_affine=False)
    non_affine_mask_layer_norm = MaskedLayerNorm(10, elementwise_affine=False)
    output_target = non_affine_layer_norm(input)
    output = non_affine_mask_layer_norm(input)

    assert torch.allclose(output, output_target, atol=1e-5)

    unset_all_random_seeds()


def test_masked_layer_norm_from_pretrained() -> None:
    layer_norm_module = nn.LayerNorm(10, elementwise_affine=True, bias=True)
    masked_layer_norm_module = MaskedLayerNorm.from_pretrained(layer_norm_module=layer_norm_module)

    assert (masked_layer_norm_module.bias is not None) and (masked_layer_norm_module.weight is not None)
    assert (not masked_layer_norm_module.weight.requires_grad) and masked_layer_norm_module.weight_scores.requires_grad
    assert (not masked_layer_norm_module.bias.requires_grad) and masked_layer_norm_module.bias_scores.requires_grad

    assert (masked_layer_norm_module.weight == layer_norm_module.weight).all()
    assert (masked_layer_norm_module.bias == layer_norm_module.bias).all()


def test_masked_batch_norm_1d() -> None:
    set_all_random_seeds(42)
    masked_batch_norm_module = MaskedBatchNorm1d(10, affine=True)
    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad

    # Test the forward pass
    input = torch.randn((3, 10))
    output = masked_batch_norm_module(input)

    # output should be equivalent to a standard layer norm with the right weights dropped
    batch_norm_module = nn.BatchNorm1d(10)
    batch_norm_module.weight = Parameter(
        torch.Tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), requires_grad=True
    )
    output_target = batch_norm_module(input)

    assert torch.allclose(output, output_target, atol=1e-5)

    non_affine_batch_norm = nn.BatchNorm1d(10, affine=False)
    non_affine_mask_batch_norm = MaskedBatchNorm1d(10, affine=False)
    output_target = non_affine_batch_norm(input)
    output = non_affine_mask_batch_norm(input)

    assert torch.allclose(output, output_target, atol=1e-5)

    unset_all_random_seeds()


def test_masked_batch_norm_1d_from_pretrained() -> None:
    batch_norm_module = nn.BatchNorm1d(10, affine=True)
    masked_batch_norm_module = MaskedBatchNorm1d.from_pretrained(batch_norm_module=batch_norm_module)

    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad

    assert (masked_batch_norm_module.weight == batch_norm_module.weight).all()
    assert (masked_batch_norm_module.bias == batch_norm_module.bias).all()


def test_masked_batch_norm_2d() -> None:
    masked_batch_norm_module = MaskedBatchNorm2d(10, affine=True)
    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad


def test_masked_batch_norm_2d_from_pretrained() -> None:
    batch_norm_module = nn.BatchNorm2d(10, affine=True)
    masked_batch_norm_module = MaskedBatchNorm2d.from_pretrained(batch_norm_module=batch_norm_module)

    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad

    assert (masked_batch_norm_module.weight == batch_norm_module.weight).all()
    assert (masked_batch_norm_module.bias == batch_norm_module.bias).all()


def test_masked_batch_norm_3d() -> None:
    masked_batch_norm_module = MaskedBatchNorm3d(10, affine=True)
    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad


def test_masked_batch_norm_3d_from_pretrained() -> None:
    batch_norm_module = nn.BatchNorm2d(10, affine=True)
    masked_batch_norm_module = MaskedBatchNorm3d.from_pretrained(batch_norm_module=batch_norm_module)

    assert (masked_batch_norm_module.bias is not None) and (masked_batch_norm_module.weight is not None)
    assert (not masked_batch_norm_module.weight.requires_grad) and masked_batch_norm_module.weight_scores.requires_grad
    assert (not masked_batch_norm_module.bias.requires_grad) and masked_batch_norm_module.bias_scores.requires_grad

    assert (masked_batch_norm_module.weight == batch_norm_module.weight).all()
    assert (masked_batch_norm_module.bias == batch_norm_module.bias).all()


def test_convert_to_masked_model() -> None:
    model1 = CompositeConvNet()
    masked_model1 = convert_to_masked_model(original_model=model1)
    assert isinstance(masked_model1.conv1d, MaskedConv1d)
    assert isinstance(masked_model1.conv2d, MaskedConv2d)
    assert isinstance(masked_model1.conv3d, MaskedConv3d)
    assert isinstance(masked_model1.linear, MaskedLinear)
    assert isinstance(masked_model1.conv_transpose1d, MaskedConvTranspose1d)
    assert isinstance(masked_model1.conv_transpose2d, MaskedConvTranspose2d)
    assert isinstance(masked_model1.conv_transpose3d, MaskedConvTranspose3d)
    assert isinstance(masked_model1.bn1d, MaskedBatchNorm1d)
    assert isinstance(masked_model1.bn2d, MaskedBatchNorm2d)
    assert isinstance(masked_model1.bn3d, MaskedBatchNorm3d)
    assert isinstance(masked_model1.layer_norm, MaskedLayerNorm)

    # Test that convert_to_masked_model properly added the score parameters
    # to all relevant modules by trying to load state_dict.
    model2 = CompositeConvNet()
    masked_model2 = convert_to_masked_model(model2)
    masked_model1.load_state_dict(masked_model2.state_dict(), strict=True)
