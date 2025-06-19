from copy import deepcopy
from logging import ERROR
from unittest.mock import Mock

import numpy as np
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, EvaluateRes, FitRes, Parameters, Scalar, Status
from flwr.server.client_manager import ClientManager, ClientProxy, SimpleClientManager
from pytest import approx, raises

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.strategies.feddg_ga import FairnessMetricType, FedDgGa
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_configure_fit_and_evaluate_success() -> None:
    fixed_sampling_client_manager = _apply_mocks_to_client_manager(FixedSamplingClientManager())
    test_n_server_rounds = 3

    def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": test_n_server_rounds,
            "evaluate_after_fit": True,
            "pack_losses_with_val_metrics": True,
        }

    def on_evaluate_config_fn(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": test_n_server_rounds,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn)
    assert strategy.num_rounds is None

    try:
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)
    except Exception as e:
        log(ERROR, "initialize_parameters threw an exception")
        raise e

    assert strategy.num_rounds == test_n_server_rounds
    assert strategy.initial_adjustment_weight == 1.0 / fixed_sampling_client_manager.num_available()
    fixed_sampling_client_manager.reset_sample.assert_called_once()  # type: ignore


def test_configure_fit_fail() -> None:
    fixed_sampling_client_manager = _apply_mocks_to_client_manager(FixedSamplingClientManager())
    simple_client_manager = _apply_mocks_to_client_manager(SimpleClientManager())

    # Fails with no configure fit
    strategy = FedDgGa()
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with bad client manager type
    def on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "evaluate_after_fit": True,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), simple_client_manager)

    # Fail with no n_server_rounds
    def on_fit_config_fn_1(server_round: int) -> dict[str, Scalar]:
        return {
            "foo": 123,
            "evaluate_after_fit": True,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_1)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with n_server_rounds not being an integer
    def on_fit_config_fn_2(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 1.1,
            "evaluate_after_fit": True,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_2)
    assert strategy.num_rounds is None

    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with evaluate_after_fit not being set
    def on_fit_config_fn_3(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_3)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with evaluate_after_fit not being True
    def on_fit_config_fn_4(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "evaluate_after_fit": False,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_4)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with pack_losses_with_val_metrics not being there
    def on_fit_config_fn_5(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "evaluate_after_fit": True,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_5)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with pack_losses_with_val_metrics not being True
    def on_fit_config_fn_6(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "evaluate_after_fit": True,
            "pack_losses_with_val_metrics": False,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_6)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)


def test_configure_evaluate_fail() -> None:
    fixed_sampling_client_manager = _apply_mocks_to_client_manager(FixedSamplingClientManager())
    simple_client_manager = _apply_mocks_to_client_manager(SimpleClientManager())

    # Fails with no evaluate fit
    strategy = FedDgGa()
    with raises(AssertionError):
        strategy.configure_evaluate(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with bad client manager type
    def on_evaluate_config_fn(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 2,
            "pack_losses_with_val_metrics": True,
        }

    strategy = FedDgGa(on_evaluate_config_fn=on_evaluate_config_fn)
    with raises(AssertionError):
        strategy.configure_evaluate(1, Parameters([], ""), simple_client_manager)

    # Fail with no pack_losses_with_val_metrics
    def on_evaluate_config_fn_1(server_round: int) -> dict[str, Scalar]:
        return {
            "foo": 123,
        }

    strategy = FedDgGa(on_evaluate_config_fn=on_evaluate_config_fn_1)
    with raises(AssertionError):
        strategy.configure_fit(1, Parameters([], ""), fixed_sampling_client_manager)

    # Fails with pack_losses_with_val_metrics not being True
    def on_fit_config_fn_2(server_round: int) -> dict[str, Scalar]:
        return {
            "n_server_rounds": 1.1,
            "pack_losses_with_val_metrics": False,
        }

    strategy = FedDgGa(on_fit_config_fn=on_fit_config_fn_2)
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
    test_initial_adjustment_weight = 1.0 / 3.0

    strategy = FedDgGa()
    strategy.num_rounds = 3
    strategy.initial_adjustment_weight = test_initial_adjustment_weight

    # test aggregate fit
    parameters_aggregated, _ = strategy.aggregate_fit(2, deepcopy(test_fit_results), [])

    assert strategy.train_metrics == {
        test_cid_1: test_fit_metrics_1,
        test_cid_2: test_fit_metrics_2,
    }
    assert strategy.adjustment_weights == {
        test_cid_1: test_initial_adjustment_weight,
        test_cid_2: test_initial_adjustment_weight,
    }
    assert parameters_aggregated is not None
    parameters_array = parameters_to_ndarrays(parameters_aggregated)[0].tolist()
    assert parameters_array == [approx(1.0, abs=0.0005), approx(1.0666, abs=0.0005)]

    # test evaluate fit
    loss_aggregated, _ = strategy.aggregate_evaluate(2, deepcopy(test_eval_results), [])

    assert strategy.evaluation_metrics == {
        test_cid_1: {**test_eval_metrics_1},
        test_cid_2: {**test_eval_metrics_2},
    }
    assert strategy.adjustment_weights == {
        test_cid_1: approx(0.2999, abs=0.0005),
        test_cid_2: approx(0.7000, abs=0.0005),
    }
    assert approx(loss_aggregated, abs=1e-6) == 1.7


def test_weight_and_aggregate_results_with_default_weights() -> None:
    test_fit_results, _ = _make_test_data()
    test_cid_1 = test_fit_results[0][0].cid
    test_cid_2 = test_fit_results[1][0].cid
    test_initial_adjustment_weight = 1.0 / 3.0

    strategy = FedDgGa()
    strategy.initial_adjustment_weight = test_initial_adjustment_weight
    aggregated_results = strategy.weight_and_aggregate_results(test_fit_results)

    assert strategy.adjustment_weights == {
        test_cid_1: test_initial_adjustment_weight,
        test_cid_2: test_initial_adjustment_weight,
    }
    assert aggregated_results[0].tolist() == [approx(1.0, abs=0.0005), approx(1.0666, abs=0.0005)]


def test_weight_and_aggregate_results_with_existing_weights() -> None:
    test_fit_results, _ = _make_test_data()
    test_cid_1 = test_fit_results[0][0].cid
    test_cid_2 = test_fit_results[1][0].cid
    test_adjustment_weights = {test_cid_1: 0.21, test_cid_2: 0.76}

    strategy = FedDgGa()
    strategy.adjustment_weights = deepcopy(test_adjustment_weights)
    aggregated_results = strategy.weight_and_aggregate_results(test_fit_results)

    assert strategy.adjustment_weights == test_adjustment_weights
    assert aggregated_results[0].tolist() == [approx(1.73, abs=0.0005), approx(1.8270, abs=0.0005)]


def test_update_weights_by_ga() -> None:
    test_cids = ["1", "2"]
    test_val_loss_key = FairnessMetricType.LOSS.value
    test_initial_adjustment_weight = 1.0 / 3.0

    strategy = FedDgGa()
    strategy.num_rounds = 3
    strategy.initial_adjustment_weight = test_initial_adjustment_weight
    strategy.train_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.evaluation_metrics = {
        test_cids[0]: {test_val_loss_key: 0.3556},
        test_cids[1]: {test_val_loss_key: 0.7654},
    }
    strategy.adjustment_weights = {
        test_cids[0]: test_initial_adjustment_weight,
        test_cids[1]: test_initial_adjustment_weight,
    }

    strategy.update_weights_by_ga(2, test_cids)

    assert strategy.adjustment_weights == {
        test_cids[0]: approx(0.2999, abs=0.0005),
        test_cids[1]: approx(0.7000, abs=0.0005),
    }


def test_update_weights_by_ga_with_same_metrics() -> None:
    test_cids = ["1", "2"]
    test_val_loss_key = FairnessMetricType.LOSS.value
    test_initial_adjustment_weight = 1.0 / 3.0

    strategy = FedDgGa()
    strategy.num_rounds = 3
    strategy.initial_adjustment_weight = test_initial_adjustment_weight
    strategy.train_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.evaluation_metrics = {
        test_cids[0]: {test_val_loss_key: 0.5467},
        test_cids[1]: {test_val_loss_key: 0.5432},
    }
    strategy.adjustment_weights = {
        test_cids[0]: test_initial_adjustment_weight,
        test_cids[1]: test_initial_adjustment_weight,
    }

    strategy.update_weights_by_ga(2, test_cids)

    assert strategy.adjustment_weights == {test_cids[0]: 0.5, test_cids[1]: 0.5}


def test_get_current_weight_step_size() -> None:
    strategy = FedDgGa()

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
    strategy.adjustment_weight_step_size = 0.5
    result_step_size = strategy.get_current_weight_step_size(6)
    assert result_step_size == approx(0.2500, abs=0.0005)


def _apply_mocks_to_client_manager(client_manager: ClientManager) -> ClientManager:
    client_proxy_1 = CustomClientProxy("1")
    client_proxy_2 = CustomClientProxy("2")
    client_manager.register(client_proxy_1)
    client_manager.register(client_proxy_2)
    client_manager.sample = Mock()  # type: ignore
    client_manager.sample.return_value = [client_proxy_1, client_proxy_2]
    client_manager.reset_sample = Mock()  # type: ignore
    return client_manager


def _make_test_data() -> tuple[list[tuple[ClientProxy, FitRes]], list[tuple[ClientProxy, EvaluateRes]]]:
    test_val_loss_key = FairnessMetricType.LOSS.value
    test_fit_metrics_1: dict[str, Scalar] = {test_val_loss_key: 1.0}
    test_fit_metrics_2: dict[str, Scalar] = {test_val_loss_key: 2.0}
    test_eval_metrics_1: dict[str, Scalar] = {"metric-1": 1.0, test_val_loss_key: 1.2}
    test_eval_metrics_2: dict[str, Scalar] = {"metric-2": 2.0, test_val_loss_key: 2.2}
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
