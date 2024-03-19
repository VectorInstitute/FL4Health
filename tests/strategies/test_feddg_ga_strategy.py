from copy import deepcopy
from typing import Dict, Union
from unittest.mock import Mock

import numpy as np
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, EvaluateRes, FitRes, Parameters, Scalar, Status
from flwr.server.client_manager import SimpleClientManager
from pytest import approx, raises

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.strategies.feddg_ga_strategy import INITIAL_WEIGHT, FairnessMetricType, FedDGGAStrategy
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_initialize_parameters_success() -> None:
    strategy = FedDGGAStrategy()
    try:
        strategy.initialize_parameters(FixedSamplingClientManager())
    except Exception as e:
        assert False, f"initialize_parameters threw an exception: {e}"


def test_initialize_parameters_fail() -> None:
    strategy = FedDGGAStrategy()
    with raises(AssertionError):
        strategy.initialize_parameters(SimpleClientManager())


def _make_mock_client_manager() -> Mock:
    mock_client_manager = Mock()
    mock_client_manager.sample.return_value = []
    mock_client_manager.num_available.return_value = 1
    return mock_client_manager


def test_configure_fit_success() -> None:

    test_n_server_rounds = 3

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "n_server_rounds": test_n_server_rounds,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn)
    assert strategy.num_rounds is None

    strategy.configure_fit(1, Parameters([], ""), _make_mock_client_manager())
    assert strategy.num_rounds == test_n_server_rounds


def test_configure_fit_fail() -> None:
    mock_client_manager = Mock()
    mock_client_manager.sample.return_value = []

    strategy = FedDGGAStrategy()
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), _make_mock_client_manager())

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "foo": 123,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), _make_mock_client_manager())

    def on_fit_config_fn_2(server_round: int) -> Dict[str, Scalar]:
        return {
            "n_server_rounds": 1.1,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn_2)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), _make_mock_client_manager())


def test_aggregate_fit_and_aggregate_evaluate() -> None:
    test_cid_1 = "1"
    test_cid_2 = "2"
    test_val_loss_key = FairnessMetricType.LOSS.value
    test_fit_metrics_1: Dict[str, Union[bool, bytes, float, int, str]] = {test_val_loss_key: 1.0}
    test_fit_metrics_2: Dict[str, Union[bool, bytes, float, int, str]] = {test_val_loss_key: 2.0}
    test_eval_metrics_1: Dict[str, Union[bool, bytes, float, int, str]] = {"metric-1": 1.0}
    test_eval_metrics_2: Dict[str, Union[bool, bytes, float, int, str]] = {"metric-2": 2.0}
    test_parameters_1 = ndarrays_to_parameters([np.array([1.0, 1.1])])
    test_parameters_2 = ndarrays_to_parameters([np.array([2.0, 2.1])])
    test_loss_1 = 1.2
    test_loss_2 = 2.2
    test_status_code = Status(Code.OK, "")
    test_num_rounds = 3
    test_fit_results = [
        (CustomClientProxy(test_cid_1), FitRes(test_status_code, test_parameters_1, 1, test_fit_metrics_1)),
        (CustomClientProxy(test_cid_2), FitRes(test_status_code, test_parameters_2, 1, test_fit_metrics_2)),
    ]
    test_evaluate_results = [
        (CustomClientProxy(test_cid_1), EvaluateRes(test_status_code, test_loss_1, 1, test_eval_metrics_1)),
        (CustomClientProxy(test_cid_2), EvaluateRes(test_status_code, test_loss_2, 1, test_eval_metrics_2)),
    ]

    strategy = FedDGGAStrategy()
    strategy.num_rounds = test_num_rounds

    # test aggregate fit
    parameters_aggregated, _ = strategy.aggregate_fit(2, deepcopy(test_fit_results), [])  # type: ignore

    assert strategy.train_metrics == {
        test_cid_1: test_fit_metrics_1,
        test_cid_2: test_fit_metrics_2,
    }
    assert strategy.adjustment_weights == {
        test_cid_1: INITIAL_WEIGHT,
        test_cid_2: INITIAL_WEIGHT,
    }
    assert parameters_aggregated is not None
    parameters_array = parameters_to_ndarrays(parameters_aggregated)[0].tolist()
    assert parameters_array == [approx(1.0, abs=0.0005), approx(1.0666, abs=0.0005)]

    # test evaluate fit
    loss_aggregated, _ = strategy.aggregate_evaluate(2, deepcopy(test_evaluate_results), [])  # type: ignore

    assert strategy.evaluation_metrics == {
        test_cid_1: {**test_eval_metrics_1, test_val_loss_key: loss_aggregated},  # type: ignore
        test_cid_2: {**test_eval_metrics_2, test_val_loss_key: loss_aggregated},  # type: ignore
    }
    assert strategy.adjustment_weights == {
        test_cid_1: approx(0.5458, abs=0.0005),
        test_cid_2: approx(0.4541, abs=0.0005),
    }


# TODO test weight_and_aggregate_results
# TODO test update_weights_by_ga
# TODO test get_current_weight_step_size
