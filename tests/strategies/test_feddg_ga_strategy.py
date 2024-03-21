from copy import deepcopy
from typing import Dict, List, Tuple, Union
from unittest.mock import Mock

import numpy as np
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, EvaluateRes, FitRes, Parameters, Scalar, Status
from flwr.server.client_manager import ClientManager, ClientProxy, SimpleClientManager
from pytest import approx, raises

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.strategies.feddg_ga_strategy import INITIAL_ADJUSTMENT_WEIGHT, FairnessMetricType, FedDGGAStrategy
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_configure_fit_success() -> None:
    fixed_sampling_client_manager = _apply_mocks_to_client_manager(FixedSamplingClientManager())
    test_n_server_rounds = 3

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "n_server_rounds": test_n_server_rounds,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn)
    assert strategy.num_rounds is None

    try:
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)
    except Exception as e:
        assert False, f"initialize_parameters threw an exception: {e}"

    assert strategy.num_rounds == test_n_server_rounds


def test_configure_fit_fail() -> None:
    fixed_sampling_client_manager = _apply_mocks_to_client_manager(FixedSamplingClientManager())
    simple_client_manager = _apply_mocks_to_client_manager(SimpleClientManager())

    strategy = FedDGGAStrategy()
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), simple_client_manager)

    def on_fit_config_fn_1(server_round: int) -> Dict[str, Scalar]:
        return {
            "foo": 123,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn_1)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    def on_fit_config_fn_2(server_round: int) -> Dict[str, Scalar]:
        return {
            "n_server_rounds": 1.1,
        }

    strategy = FedDGGAStrategy(on_fit_config_fn=on_fit_config_fn_2)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)


def test_aggregate_fit_and_aggregate_evaluate() -> None:
    test_fit_results, test_eval_results = _make_test_data()
    test_cid_1 = test_fit_results[0][0].cid
    test_cid_2 = test_fit_results[1][0].cid
    test_fit_metrics_1 = test_fit_results[0][1].metrics
    test_fit_metrics_2 = test_fit_results[1][1].metrics
    test_eval_metrics_1 = test_eval_results[0][1].metrics
    test_eval_metrics_2 = test_eval_results[1][1].metrics
    test_val_loss_key = FairnessMetricType.LOSS.value

    strategy = FedDGGAStrategy()
    strategy.num_rounds = 3

    # test aggregate fit
    parameters_aggregated, _ = strategy.aggregate_fit(2, deepcopy(test_fit_results), [])

    assert strategy.train_metrics == {
        test_cid_1: test_fit_metrics_1,
        test_cid_2: test_fit_metrics_2,
    }
    assert strategy.adjustment_weights == {
        test_cid_1: INITIAL_ADJUSTMENT_WEIGHT,
        test_cid_2: INITIAL_ADJUSTMENT_WEIGHT,
    }
    assert parameters_aggregated is not None
    parameters_array = parameters_to_ndarrays(parameters_aggregated)[0].tolist()
    assert parameters_array == [approx(1.0, abs=0.0005), approx(1.0666, abs=0.0005)]

    # test evaluate fit
    loss_aggregated, _ = strategy.aggregate_evaluate(2, deepcopy(test_eval_results), [])

    assert strategy.evaluation_metrics == {
        test_cid_1: {**test_eval_metrics_1, test_val_loss_key: loss_aggregated},
        test_cid_2: {**test_eval_metrics_2, test_val_loss_key: loss_aggregated},
    }
    assert strategy.adjustment_weights == {
        test_cid_1: approx(0.5458, abs=0.0005),
        test_cid_2: approx(0.4541, abs=0.0005),
    }


def test_weight_and_aggregate_results_with_default_weights() -> None:
    test_fit_results, _ = _make_test_data()
    test_cid_1 = test_fit_results[0][0].cid
    test_cid_2 = test_fit_results[1][0].cid

    strategy = FedDGGAStrategy()
    aggregated_results = strategy.weight_and_aggregate_results(test_fit_results)

    assert strategy.adjustment_weights == {
        test_cid_1: INITIAL_ADJUSTMENT_WEIGHT,
        test_cid_2: INITIAL_ADJUSTMENT_WEIGHT,
    }
    assert aggregated_results[0].tolist() == [approx(1.0, abs=0.0005), approx(1.0666, abs=0.0005)]


def test_weight_and_aggregate_results_with_existing_weights() -> None:
    test_fit_results, _ = _make_test_data()
    test_cid_1 = test_fit_results[0][0].cid
    test_cid_2 = test_fit_results[1][0].cid
    test_adjustment_weights = {test_cid_1: 0.21, test_cid_2: 0.76}

    strategy = FedDGGAStrategy()
    strategy.adjustment_weights = deepcopy(test_adjustment_weights)
    aggregated_results = strategy.weight_and_aggregate_results(test_fit_results)

    assert strategy.adjustment_weights == test_adjustment_weights
    assert aggregated_results[0].tolist() == [approx(1.73, abs=0.0005), approx(1.8270, abs=0.0005)]


def test_update_weights_by_ga() -> None:
    test_cids = ["1", "2"]
    test_val_loss_key = FairnessMetricType.LOSS.value

    strategy = FedDGGAStrategy()
    strategy.num_rounds = 3
    strategy.train_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.evaluation_metrics = {
        test_cids[0]: {test_val_loss_key: 0.3556},
        test_cids[1]: {test_val_loss_key: 0.7654},
    }
    strategy.adjustment_weights = {
        test_cids[0]: INITIAL_ADJUSTMENT_WEIGHT,
        test_cids[1]: INITIAL_ADJUSTMENT_WEIGHT,
    }

    strategy.update_weights_by_ga(2, test_cids)

    assert strategy.adjustment_weights == {
        test_cids[0]: approx(0.4385, abs=0.0005),
        test_cids[1]: approx(0.5614, abs=0.0005),
    }


def test_update_weights_by_ga_with_same_metrics() -> None:
    test_cids = ["1", "2"]
    test_val_loss_key = FairnessMetricType.LOSS.value

    strategy = FedDGGAStrategy()
    strategy.num_rounds = 3
    strategy.train_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.evaluation_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.adjustment_weights = {
        test_cids[0]: INITIAL_ADJUSTMENT_WEIGHT,
        test_cids[1]: INITIAL_ADJUSTMENT_WEIGHT,
    }

    strategy.update_weights_by_ga(2, test_cids)

    assert strategy.adjustment_weights == {test_cids[0]: 0.5, test_cids[1]: 0.5}


def test_get_current_weight_step_size() -> None:
    strategy = FedDGGAStrategy()

    with raises(AssertionError):
        strategy.get_current_weight_step_size(2)

    strategy.num_rounds = 3
    result_step_size = strategy.get_current_weight_step_size(1)
    assert result_step_size == approx(0.2000, abs=0.0005)
    result_step_size = strategy.get_current_weight_step_size(2)
    assert result_step_size == approx(0.1333, abs=0.0005)
    result_step_size = strategy.get_current_weight_step_size(3)
    assert result_step_size == approx(0.0666, abs=0.0005)

    strategy.num_rounds = 10
    result_step_size = strategy.get_current_weight_step_size(6)
    assert result_step_size == approx(0.1000, abs=0.0005)

    strategy.num_rounds = 10
    strategy.weight_step_size = 0.5
    result_step_size = strategy.get_current_weight_step_size(6)
    assert result_step_size == approx(0.2500, abs=0.0005)


def _apply_mocks_to_client_manager(client_manager: ClientManager) -> ClientManager:
    client_proxy = CustomClientProxy("1")
    client_manager.register(client_proxy)
    client_manager.sample = Mock()  # type: ignore
    client_manager.sample.return_value = [client_proxy]
    return client_manager


def _make_test_data() -> Tuple[List[Tuple[ClientProxy, FitRes]], List[Tuple[ClientProxy, EvaluateRes]]]:
    test_val_loss_key = FairnessMetricType.LOSS.value
    test_fit_metrics_1: Dict[str, Union[bool, bytes, float, int, str]] = {test_val_loss_key: 1.0}
    test_fit_metrics_2: Dict[str, Union[bool, bytes, float, int, str]] = {test_val_loss_key: 2.0}
    test_eval_metrics_1: Dict[str, Union[bool, bytes, float, int, str]] = {"metric-1": 1.0}
    test_eval_metrics_2: Dict[str, Union[bool, bytes, float, int, str]] = {"metric-2": 2.0}
    test_parameters_1 = ndarrays_to_parameters([np.array([1.0, 1.1])])
    test_parameters_2 = ndarrays_to_parameters([np.array([2.0, 2.1])])
    test_fit_results = [
        (CustomClientProxy("1"), FitRes(Status(Code.OK, ""), test_parameters_1, 1, test_fit_metrics_1)),
        (CustomClientProxy("2"), FitRes(Status(Code.OK, ""), test_parameters_2, 1, test_fit_metrics_2)),
    ]
    test_evaluate_results = [
        (CustomClientProxy("1"), EvaluateRes(Status(Code.OK, ""), 1.2, 1, test_eval_metrics_1)),
        (CustomClientProxy("2"), EvaluateRes(Status(Code.OK, ""), 2.2, 1, test_eval_metrics_2)),
    ]

    return test_fit_results, test_evaluate_results  # type: ignore
