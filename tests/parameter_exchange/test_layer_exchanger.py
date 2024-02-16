import numpy as np
import torch
import torch.nn as nn

from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger, FixedLayerExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import LayerSelectionFunctionConstructor
from tests.test_utils.models_for_test import LinearModel, ToyConvNet


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
