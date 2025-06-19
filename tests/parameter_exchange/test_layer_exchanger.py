import numpy as np
import torch
from torch import nn

from fl4health.model_bases.masked_layers.masked_layers_utils import convert_to_masked_model
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger, FixedLayerExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import (
    LayerSelectionFunctionConstructor,
    select_scores_and_sample_masks,
)
from fl4health.utils.functions import sigmoid_inverse
from tests.test_utils.models_for_test import CompositeConvNet, LinearModel, ModelWrapper, ToyConvNet


def test_layer_parameter_exchange() -> None:
    model = LinearModel()
    # fill model weights with different constants
    nn.init.constant_(model.independent_layer.weight, 1.0)
    nn.init.constant_(model.shared_layer.weight, 2.0)
    exchanger = FixedLayerExchanger(["shared_layer.weight"])
    shared_layer_list = exchanger.push_parameters(model)
    assert len(shared_layer_list) == 1
    shared_layer_exchanged = shared_layer_list[0]
    assert np.array_equal(shared_layer_exchanged, 2.0 * np.ones((3, 3)))
    exchanger.pull_parameters([2.0 * p for p in shared_layer_list], model)
    assert torch.all(torch.eq(model.shared_layer.weight, 4.0 * torch.ones((3, 3))))
    assert torch.all(torch.eq(model.independent_layer.weight, torch.ones((4, 4))))


def test_norm_drift_layer_exchange() -> None:
    initial_model = ToyConvNet()
    model_to_exchange = ToyConvNet()

    nn.init.constant_(initial_model.conv1.weight, 0)
    nn.init.constant_(initial_model.conv2.weight, 0)
    nn.init.constant_(initial_model.fc1.weight, 0)
    nn.init.constant_(initial_model.fc2.weight, 0)

    nn.init.constant_(model_to_exchange.conv1.weight, 1 / 20000)
    nn.init.constant_(model_to_exchange.conv2.weight, 1 / 20000)
    nn.init.constant_(model_to_exchange.fc1.weight, 1)
    nn.init.constant_(model_to_exchange.fc2.weight, 2)

    norm_threshold = 2
    exchange_percentage = 0.5

    selection_function_constructor = LayerSelectionFunctionConstructor(
        norm_threshold=norm_threshold,
        exchange_percentage=exchange_percentage,
        normalize=False,
        select_drift_more=True,
    )

    select_layers_by_threshold_func = selection_function_constructor.select_by_threshold()

    exchanger = DynamicLayerExchanger(
        layer_selection_function=select_layers_by_threshold_func,
    )

    # Test selecting layers by thresholding their drift norms.
    # Assume we do not normalize by the number of parameters
    # when calculating a tensor's drift norm and
    # select layers that drift the most.
    layers_with_names_to_exchange = exchanger.push_parameters(model_to_exchange, initial_model)
    layers_to_exchange, layer_names = exchanger.unpack_parameters(layers_with_names_to_exchange)
    assert len(layers_to_exchange) == 2
    assert len(layer_names) == 2

    # Normalize when calculating drift norm.
    selection_function_constructor.normalize = True
    select_layers_by_threshold_func_normalize = selection_function_constructor.select_by_threshold()
    exchanger.layer_selection_function = select_layers_by_threshold_func_normalize
    layers_with_names_to_exchange = exchanger.push_parameters(model_to_exchange, initial_model)
    layers_to_exchange, layer_names = exchanger.unpack_parameters(layers_with_names_to_exchange)
    assert len(layers_to_exchange) == 0
    assert len(layer_names) == 0

    # Select the layers that drift the most in terms of l2 norm,
    # where the number of layers selected is determined by exchange_percentage.
    select_layers_by_percentage_func = selection_function_constructor.select_by_percentage()
    exchanger.layer_selection_function = select_layers_by_percentage_func
    layers_with_names_to_exchange = exchanger.push_parameters(model_to_exchange, initial_model)
    layers_to_exchange, layer_names = exchanger.unpack_parameters(layers_with_names_to_exchange)
    assert len(layer_names) == 2
    assert len(layers_to_exchange) == 2


def test_fedpm_exchange() -> None:
    model = CompositeConvNet()
    wrapped_model = ModelWrapper(model)

    masked_model = convert_to_masked_model(model)
    masked_wrapped_model = convert_to_masked_model(wrapped_model)

    masked_model_states = masked_model.state_dict()
    wrapped_model_states = masked_wrapped_model.state_dict()

    # Test that selection function works when the direct child modules are masked modules.
    masks, score_names = select_scores_and_sample_masks(masked_model, masked_model)
    assert len(masks) == len(score_names)
    assert score_names == [
        "conv1d.weight_scores",
        "conv1d.bias_scores",
        "bn1d.weight_scores",
        "bn1d.bias_scores",
        "conv2d.weight_scores",
        "conv2d.bias_scores",
        "bn2d.weight_scores",
        "bn2d.bias_scores",
        "conv3d.weight_scores",
        "conv3d.bias_scores",
        "bn3d.weight_scores",
        "bn3d.bias_scores",
        "conv_transpose1d.weight_scores",
        "conv_transpose1d.bias_scores",
        "conv_transpose2d.weight_scores",
        "conv_transpose2d.bias_scores",
        "conv_transpose3d.weight_scores",
        "conv_transpose3d.bias_scores",
        "linear.weight_scores",
        "linear.bias_scores",
        "layer_norm.weight_scores",
        "layer_norm.bias_scores",
    ]
    for i in range(len(score_names)):
        mask = masks[i]
        score_name = score_names[i]
        assert mask.shape == masked_model_states[score_name].cpu().numpy().shape
        assert ((mask == 0) | (mask == 1)).all()

    # Test that the selection function works when there are masked modules which are not direct child modules.
    wrapped_model_states = masked_wrapped_model.state_dict()
    masks_wrapped, score_names_wrapped = select_scores_and_sample_masks(masked_wrapped_model, masked_wrapped_model)
    assert len(masks_wrapped) == len(score_names_wrapped)
    assert score_names_wrapped == [
        "model.conv1d.weight_scores",
        "model.conv1d.bias_scores",
        "model.bn1d.weight_scores",
        "model.bn1d.bias_scores",
        "model.conv2d.weight_scores",
        "model.conv2d.bias_scores",
        "model.bn2d.weight_scores",
        "model.bn2d.bias_scores",
        "model.conv3d.weight_scores",
        "model.conv3d.bias_scores",
        "model.bn3d.weight_scores",
        "model.bn3d.bias_scores",
        "model.conv_transpose1d.weight_scores",
        "model.conv_transpose1d.bias_scores",
        "model.conv_transpose2d.weight_scores",
        "model.conv_transpose2d.bias_scores",
        "model.conv_transpose3d.weight_scores",
        "model.conv_transpose3d.bias_scores",
        "model.linear.weight_scores",
        "model.linear.bias_scores",
        "model.layer_norm.weight_scores",
        "model.layer_norm.bias_scores",
    ]
    for j in range(len(score_names_wrapped)):
        mask = masks_wrapped[j]
        score_name = score_names_wrapped[j]
        assert mask.shape == wrapped_model_states[score_name].cpu().numpy().shape

    # Test that pull_parameter works as expected.
    parameter_exchanger = FedPmExchanger()
    packed_parameters = parameter_exchanger.pack_parameters(model_weights=masks, additional_parameters=score_names)
    masked_model_new = convert_to_masked_model(CompositeConvNet())
    parameter_exchanger.pull_parameters(packed_parameters, masked_model_new)
    for i in range(len(score_names)):
        mask = masks[i]
        score_name = score_names[i]
        prob_scores = sigmoid_inverse(torch.tensor(mask))
        assert (prob_scores == masked_model_new.state_dict()[score_name]).all()
