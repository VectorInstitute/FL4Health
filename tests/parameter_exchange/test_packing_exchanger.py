from typing import List

import numpy as np
import pytest
from flwr.common.typing import NDArrays

from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerWithClippingBit,
    ParameterPackerWithControlVariates,
    ParameterPackerWithExtraVariables,
    ParameterPackerWithLayerNames,
)


@pytest.fixture
def get_ndarrays(layer_sizes: List[List[int]]) -> NDArrays:
    ndarrays = [np.ones(tuple(size)) for size in layer_sizes]
    return ndarrays


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


@pytest.mark.parametrize("layer_sizes", [[[3, 3] for _ in range(6)]])
def test_parameter_packer_with_extra_variables(get_ndarrays: NDArrays) -> None:  # noqa
    model_weights = get_ndarrays  # noqa
    extra_variables = {}
    extra_variables["var1"] = [np.array(0.0)]

    packer = ParameterPackerWithExtraVariables(len(model_weights))

    packed_params = packer.pack_parameters(model_weights, extra_variables)

    assert len(packed_params) == len(model_weights) + 1 + sum(
        [len(extra_variable) for extra_variable in extra_variables.values()]
    )

    correct_packed_params = model_weights + [np.array(list(extra_variables.keys()))]
    for _, values in extra_variables.items():
        correct_packed_params = correct_packed_params + values

    for packed_param, correct_packed_param in zip(packed_params, correct_packed_params):
        assert packed_param.size == correct_packed_param.size

    unpacked_model_weights, unpacked_extra_variables = packer.unpack_parameters(packed_params)

    for model_weight, unpacked_model_weight in zip(model_weights, unpacked_model_weights):
        assert model_weight.size == unpacked_model_weight.size

    assert extra_variables == unpacked_extra_variables
    assert len(extra_variables) == len(unpacked_extra_variables)
