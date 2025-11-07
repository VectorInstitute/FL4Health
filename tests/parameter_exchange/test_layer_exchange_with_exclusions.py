import torch
from torch import nn

from fl4health.parameter_exchange.layer_exchanger import LayerExchangerWithExclusions
from tests.test_utils.models_for_test import ToyConvNet, UNet3D


def test_simple_model_exclusion() -> None:
    model = ToyConvNet(include_bn=True)

    conv1_weights = model.conv1.weight.detach().clone()
    conv2_weights = model.conv2.weight.detach().clone()
    bn1_weights = model.bn1.weight.detach().clone()
    # fill model weights with different constants
    nn.init.constant_(model.fc1.weight, 1.0)
    nn.init.constant_(model.fc2.weight, 2.0)
    exchanger = LayerExchangerWithExclusions(model, {nn.Conv2d, nn.BatchNorm1d})
    shared_layer_list = exchanger.push_parameters(model)

    # Note that Max pool has no trainable parameters and is therefore not exchanged.
    assert len(shared_layer_list) == 2

    # Modify the weights before putting them back in the module.
    exchanger.pull_parameters([2.0 * p for p in shared_layer_list], model)

    # Excluded weights should be the same
    assert torch.all(torch.eq(model.conv1.weight, conv1_weights))
    assert torch.all(torch.eq(model.conv2.weight, conv2_weights))
    assert torch.all(torch.eq(model.bn1.weight, bn1_weights))
    assert torch.all(torch.eq(model.fc1.weight, 2.0 * torch.ones((120, 16 * 4 * 4))))
    assert torch.all(torch.eq(model.fc2.weight, 4.0 * torch.ones((64, 120))))


def test_nested_model_exclusion() -> None:
    model = UNet3D(num_encoding_blocks=2, out_channels_first_layer=2)

    bn_weights = model.encoder.encoding_blocks[0].conv2.norm_layer.weight.detach().clone()  # type: ignore
    # fill model weights with different constants
    exchanger = LayerExchangerWithExclusions(model, {nn.BatchNorm3d})
    shared_layer_list = exchanger.push_parameters(model)

    # Note that Max pool has no trainable parameters and is therefore not exchanged.
    assert len(shared_layer_list) == 30

    # Modify the weights to be zero before putting them back in the module.
    exchanger.pull_parameters([0.0 * p for p in shared_layer_list], model)

    # Excluded weights should be the same as before, these two weights are actually tied together in the U-net
    assert torch.all(torch.eq(model.encoder.encoding_blocks[0].conv2.norm_layer.weight, bn_weights))  # type: ignore
    assert torch.all(torch.eq(model.encoder.encoding_blocks[0].conv2.block[1].weight, bn_weights))  # type: ignore

    # These weights should be zero, as they were "exchanged"
    weights = model.decoder.decoding_blocks[0].conv2.conv_layer.weight  # type: ignore
    assert isinstance(weights, torch.Tensor)
    assert torch.all(torch.eq(weights, torch.zeros_like(weights)))
