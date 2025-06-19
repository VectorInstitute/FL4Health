import numpy as np
import pytest
from flwr.common import parameters_to_ndarrays

from fl4health.strategies.basic_fedavg import OpacusBasicFedAvg
from fl4health.strategies.scaffold import OpacusScaffold
from fl4health.utils.privacy_utilities import map_model_to_opacus_model
from tests.test_utils.models_for_test import MnistNetWithBnAndFrozen


model_for_tests = MnistNetWithBnAndFrozen(True)


def test_opacus_basic_fedavg() -> None:
    # This should throw AttributeError: Provided model must be Opacus type GradSampleModule
    with pytest.raises(AssertionError) as assertion_exception:
        strategy = OpacusBasicFedAvg(model=model_for_tests)

    assert str(assertion_exception.value) == "Provided model must be Opacus type GradSampleModule"

    opacus_model = map_model_to_opacus_model(model_for_tests)
    # Should successfully create strategy without throwing now
    strategy = OpacusBasicFedAvg(model=opacus_model)
    assert strategy.initial_parameters is not None
    strategy_initial_arrays = parameters_to_ndarrays(strategy.initial_parameters)

    model_arrays = [val.cpu().numpy() for _, val in opacus_model.state_dict().items()]
    assert len(model_arrays) > 0
    assert len(strategy_initial_arrays) > 0
    for model_array, strategy_array in zip(model_arrays, strategy_initial_arrays):
        assert np.allclose(model_array, strategy_array, atol=0.0001)


def test_opacus_scaffold() -> None:
    # This should throw AttributeError: Provided model must be Opacus type GradSampleModule
    with pytest.raises(AssertionError) as assertion_exception:
        strategy = OpacusScaffold(model=model_for_tests)

    assert str(assertion_exception.value) == "Provided model must be Opacus type GradSampleModule"

    opacus_model = map_model_to_opacus_model(model_for_tests)
    # Should successfully create strategy without throwing now
    strategy = OpacusScaffold(model=opacus_model)
    assert strategy.initial_parameters is not None
    strategy_initial_arrays, strategy_variates_arrays = strategy.parameter_packer.unpack_parameters(
        parameters_to_ndarrays(strategy.initial_parameters)
    )

    model_arrays = [val.cpu().numpy() for _, val in opacus_model.state_dict().items()]
    model_requires_grad_params = [np.zeros_like(val.data) for val in opacus_model.parameters() if val.requires_grad]
    assert len(model_arrays) > 0
    assert len(strategy_initial_arrays) > 0
    for model_array, strategy_array in zip(model_arrays, strategy_initial_arrays):
        assert np.allclose(model_array, strategy_array, atol=0.0001)

    assert len(model_requires_grad_params) > 0
    assert len(strategy_variates_arrays) > 0
    for model_array, strategy_array in zip(model_requires_grad_params, strategy_variates_arrays):
        assert np.allclose(model_array, strategy_array, atol=0.0001)
