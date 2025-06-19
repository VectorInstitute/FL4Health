import copy

import numpy as np
import pytest
import torch
from flwr.common.typing import NDArrays
from torch import Tensor

from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
    ParameterPackerWithClippingBit,
    ParameterPackerWithControlVariates,
    ParameterPackerWithLayerNames,
    SparseCooParameterPacker,
)
from fl4health.parameter_exchange.parameter_selection_criteria import (
    largest_final_magnitude_scores,
    largest_increase_in_magnitude_scores,
    largest_magnitude_change_scores,
    smallest_final_magnitude_scores,
    smallest_increase_in_magnitude_scores,
    smallest_magnitude_change_scores,
)
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import SparseCooParameterExchanger
from tests.test_utils.models_for_test import ConstantConvNet, ToyConvNet


@pytest.fixture
def get_ndarrays(layer_sizes: list[list[int]]) -> NDArrays:
    return [np.ones(tuple(size)) for size in layer_sizes]


@pytest.fixture
def get_sparse_tensors(num_tensors: int) -> list[Tensor]:
    tensors = []
    for _ in range(num_tensors):
        x = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [5, 0, 0, 0]])
        tensors.append(x)
    return tensors


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_exchanger_with_control_variates(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    control_variates = get_ndarrays  # noqa

    exchanger = FullParameterExchangerWithPacking(ParameterPackerWithControlVariates(len(model_weights)))
    packed_params = exchanger.pack_parameters(model_weights, control_variates)

    assert len(packed_params) == len(model_weights) + len(control_variates)

    correct_packed_params = model_weights + control_variates
    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert packed_param.size == correct_packed_param.size

    unpacked_model_weights, unpacked_control_variates = exchanger.unpack_parameters(packed_params)

    assert len(unpacked_model_weights) == len(model_weights)
    assert len(unpacked_control_variates) == len(control_variates)

    for model_weight, unpacked_model_weight in zip(model_weights, unpacked_model_weights):
        assert model_weight.size == unpacked_model_weight.size

    for control_variate, unpacked_control_variate in zip(control_variates, unpacked_control_variates):
        assert control_variate.size == unpacked_control_variate.size


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_exchanger_with_clipping_bits(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    clipping_bit = 0.0

    exchanger = FullParameterExchangerWithPacking(ParameterPackerWithClippingBit())

    packed_params = exchanger.pack_parameters(model_weights, clipping_bit)

    assert len(packed_params) == len(model_weights) + 1

    correct_packed_params = model_weights + [np.array(clipping_bit)]

    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert packed_param.size == correct_packed_param.size

    unpacked_model_weights, unpacked_clipping_bit = exchanger.unpack_parameters(packed_params)

    for model_weight, unpacked_model_weight in zip(model_weights, unpacked_model_weights):
        assert model_weight.size == unpacked_model_weight.size

    assert clipping_bit == unpacked_clipping_bit


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_exchanger_fedprox(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    extra_fedprox_variable = 0.0

    exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    packed_params = exchanger.pack_parameters(model_weights, extra_fedprox_variable)

    assert len(packed_params) == len(model_weights) + 1

    correct_packed_params = model_weights + [np.array(extra_fedprox_variable)]

    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert packed_param.size == correct_packed_param.size

    unpacked_model_weights, unpacked_extra_fedprox_variable = exchanger.unpack_parameters(packed_params)

    for model_weight, unpacked_model_weight in zip(model_weights, unpacked_model_weights):
        assert model_weight.size == unpacked_model_weight.size

    assert extra_fedprox_variable == unpacked_extra_fedprox_variable


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_packer_with_layer_names(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    weights_names = ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6"]

    packer = ParameterPackerWithLayerNames()

    packed_params = packer.pack_parameters(model_weights, weights_names)

    assert len(packed_params) == len(model_weights) + 1

    correct_packed_params = model_weights + [np.array(weights_names)]

    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert packed_param.size == correct_packed_param.size

    unpacked_model_weights, unpacked_weights_names = packer.unpack_parameters(packed_params)

    for model_weight, unpacked_model_weight in zip(model_weights, unpacked_model_weights):
        assert model_weight.size == unpacked_model_weight.size

    assert weights_names == unpacked_weights_names
    assert len(weights_names) == len(model_weights)


@pytest.mark.parametrize("num_tensors", [6])
def test_sparse_coo_parameter_packer(get_sparse_tensors: list[Tensor]) -> None:
    model_tensors = get_sparse_tensors
    tensor_names = ["tensor1", "tensor2", "tensor3", "tensor4", "tensor5", "tensor6"]
    parameter_nonzero_values = []
    parameter_indices = []
    tensor_shapes = []

    for tensor in model_tensors:
        nonzero_values = tensor[torch.nonzero(tensor, as_tuple=True)]
        nonzero_indices = torch.nonzero(tensor, as_tuple=False)
        tensor_shape = np.array(list(tensor.shape))

        parameter_nonzero_values.append(nonzero_values.numpy())
        parameter_indices.append(nonzero_indices.numpy())
        tensor_shapes.append(tensor_shape)

    packer = SparseCooParameterPacker()

    packed_params = packer.pack_parameters(
        model_parameters=parameter_nonzero_values,
        additional_parameters=(parameter_indices, tensor_shapes, tensor_names),
    )

    assert len(packed_params) == 3 * len(parameter_nonzero_values) + 1

    correct_packed_params = parameter_nonzero_values + parameter_indices + tensor_shapes + [np.array(tensor_names)]

    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert (packed_param == correct_packed_param).all()

    unpacked_param_nonzero_values, unpacked_additional_info = packer.unpack_parameters(packed_params)

    unpacked_parameter_indices, unpacked_tensor_shapes, unpacked_tensor_names = unpacked_additional_info

    assert unpacked_parameter_indices == parameter_indices
    assert unpacked_tensor_shapes == tensor_shapes
    assert unpacked_tensor_names == tensor_names
    assert unpacked_param_nonzero_values == parameter_nonzero_values


def test_sparse_coo_parameter_exchanger_sparsity_level() -> None:
    torch.manual_seed(42)
    toy_convnet = ToyConvNet(include_bn=False)
    toy_convnet_initial = ToyConvNet(include_bn=False)

    parameter_exchanger = SparseCooParameterExchanger(
        sparsity_level=0.1, score_gen_function=largest_final_magnitude_scores
    )

    total_params_number = sum(p.numel() for p in toy_convnet.parameters())

    # Test that the parameter exchanger only selects 10 percent of the parameters.
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(toy_convnet, toy_convnet_initial)
    selected_params_number = sum(len(t) for t in nonzero_vals)
    assert selected_params_number * 10 == total_params_number
    indices, _, _ = additional_parameters
    for non_zero_vals_one_tensor, indices_one_tensor in zip(nonzero_vals, indices):
        assert len(non_zero_vals_one_tensor) == len(indices_one_tensor)


def test_sparse_coo_parameter_exchanger() -> None:
    initial_model_const = ConstantConvNet(constants=[0.1, 2, 3.2, 5])
    model_const = ConstantConvNet(constants=[1.5, 2.5, 3.5, 4.5])

    parameter_exchanger = SparseCooParameterExchanger(
        sparsity_level=0.001, score_gen_function=largest_final_magnitude_scores
    )

    # Test parameter selection (with the largest_final_magnitude_scores as criterion).
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 4.5).all()
    assert len(indices[0]) == 64
    assert (shapes[0] == np.array([4, 16])).all()
    assert tensor_names[0] == "fc2.weight"

    # Test parameter loading
    new_nonzero_vals = [np.full(shape=nonzero_vals[0].shape, fill_value=6.6)]

    packed_parameters = parameter_exchanger.pack_parameters(
        model_weights=new_nonzero_vals, additional_parameters=(indices, shapes, tensor_names)
    )

    model_copy = copy.deepcopy(model_const)

    parameter_exchanger.pull_parameters(packed_parameters, model_copy)

    assert (model_copy.conv1.weight == 1.5).all()
    assert (model_copy.conv2.weight == 2.5).all()
    assert (model_copy.fc1.weight == 3.5).all()
    assert (model_copy.fc2.weight == 6.6).all()

    # Test parameter selection with other criteria
    parameter_exchanger.score_gen_function = largest_magnitude_change_scores
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 1.5).all()
    assert len(indices[0]) == 150
    assert (shapes[0] == np.array([6, 1, 5, 5])).all()
    assert tensor_names[0] == "conv1.weight"

    parameter_exchanger.score_gen_function = largest_increase_in_magnitude_scores
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 1.5).all()
    assert len(indices[0]) == 150
    assert (shapes[0] == np.array([6, 1, 5, 5])).all()
    assert tensor_names[0] == "conv1.weight"

    parameter_exchanger.score_gen_function = smallest_final_magnitude_scores
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 1.5).all()
    assert len(indices[0]) == 150
    assert (shapes[0] == np.array([6, 1, 5, 5])).all()
    assert tensor_names[0] == "conv1.weight"

    parameter_exchanger.score_gen_function = smallest_magnitude_change_scores
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 3.5).all()
    assert len(indices[0]) == 4096
    assert (shapes[0] == np.array([16, 256])).all()
    assert tensor_names[0] == "fc1.weight"

    parameter_exchanger.score_gen_function = smallest_increase_in_magnitude_scores
    nonzero_vals, additional_parameters = parameter_exchanger.select_parameters(model_const, initial_model_const)
    indices, shapes, tensor_names = additional_parameters
    assert len(nonzero_vals) == 1 and len(indices) == 1 and len(shapes) == 1 and len(tensor_names) == 1
    assert (nonzero_vals[0] == 4.5).all()
    assert len(indices[0]) == 64
    assert (shapes[0] == np.array([4, 16])).all()
    assert tensor_names[0] == "fc2.weight"
