import copy
from typing import List

import numpy as np
import pytest
import torch
from flwr.common.typing import NDArrays
from torch import Tensor

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerFedProx,
    ParameterPackerWithClippingBit,
    ParameterPackerWithControlVariates,
    ParameterPackerWithLayerNames,
    SparseCooParameterPacker,
)
from fl4health.parameter_exchange.sparse_coo_parameter_exchanger import (
    SparseCooParameterExchanger,
    largest_final_magnitude_scores,
)
from tests.test_utils.models_for_test import ConstantConvNet


@pytest.fixture
def get_ndarrays(layer_sizes: List[List[int]]) -> NDArrays:
    ndarrays = [np.ones(tuple(size)) for size in layer_sizes]
    return ndarrays


@pytest.fixture
def get_sparse_tensors(num_tensors: int) -> List[Tensor]:
    tensors = []
    for _ in range(num_tensors):
        x = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [5, 0, 0, 0]])
        tensors.append(x)
    return tensors


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_exchanger_with_control_variates(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    control_variates = get_ndarrays  # noqa

    exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(len(model_weights)))
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

    exchanger = ParameterExchangerWithPacking(ParameterPackerWithClippingBit())

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

    exchanger = ParameterExchangerWithPacking(ParameterPackerFedProx())

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
def test_sparse_coo_parameter_packer(get_sparse_tensors: List[Tensor]) -> None:
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


def test_sparse_coo_parameter_exchanger() -> None:
    initial_model = ConstantConvNet(constants=[0.1, 2, 3.2, 5])
    model = ConstantConvNet(constants=[1.5, 2.5, 3.5, 4.5])

    parameter_exchanger = SparseCooParameterExchanger(
        sparsity_level=0.1, score_gen_function=largest_final_magnitude_scores
    )

    # Test parameter selection.
    nonzero_vals, indices, shapes, tensor_names = parameter_exchanger.select_parameters(model, initial_model)
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

    model_copy = copy.deepcopy(model)

    parameter_exchanger.pull_parameters(packed_parameters, model_copy)

    assert (model_copy.conv1.weight == 1.5).all()
    assert (model_copy.conv2.weight == 2.5).all()
    assert (model_copy.fc1.weight == 3.5).all()
    assert (model_copy.fc2.weight == 6.6).all()
