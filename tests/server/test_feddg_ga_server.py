from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import numpy as np
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Code, EvaluateRes, FitRes, Parameters, Scalar, Status
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.strategy.fedavg import FedAvg
from pytest import approx

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.server.feddg_ga_server import ClientMetrics, FairnessMetricType, FedDGGAServer
from tests.test_utils.custom_client_proxy import CustomClientProxy


@patch("fl4health.server.feddg_ga_server.FlServer.fit")
def test_fit(mock_super_fit: Mock) -> None:
    test_num_rounds = 3
    server = FedDGGAServer()

    assert server.num_rounds is None
    server.fit(test_num_rounds, None)
    mock_super_fit.assert_called_once_with(test_num_rounds, None)
    assert server.num_rounds == test_num_rounds


@patch("fl4health.server.feddg_ga_server.FlServer.evaluate_round")
@patch("fl4health.server.feddg_ga_server.FlServer.fit_round")
def test_fit_and_evaluate_round(mock_super_fit_round: Mock, mock_super_evaluate_round: Mock) -> None:
    # Setting up
    test_total_rounds = 5
    test_server_round = 2
    test_num_clients = 2
    test_loss = 0.1
    test_fairness_metric = FairnessMetricType.LOSS.value
    test_fit_metrics = [
        {"test_fit_metric": 1.1, test_fairness_metric: 1.2},
        {"test_fit_metric": 2.1, test_fairness_metric: 2.2},
    ]
    test_fit_results = [
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, test_fit_metrics[0])],  # type: ignore
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, test_fit_metrics[1])],  # type: ignore
    ]
    test_evaluate_results = [
        [None, EvaluateRes(Status(Code.OK, ""), 3.0, 2, {"test_eval_metric": 4.0})],
        [None, EvaluateRes(Status(Code.OK, ""), 5.0, 2, {"test_eval_metric": 6.0})],
    ]
    test_failures: List[Any] = []
    test_fit_results_and_failures = (test_fit_results, test_failures)
    test_evaluate_results_and_failures = (test_evaluate_results, test_failures)
    server = _make_server(test_num_clients)
    server.num_rounds = test_total_rounds
    _setup_server_fit_and_evaluate_mocks(
        server=server,
        loss=test_loss,
        fit_results_and_failures=deepcopy(test_fit_results_and_failures),
        evaluate_results_and_failures=deepcopy(test_evaluate_results_and_failures),
        mock_fit_round=mock_super_fit_round,
        mock_evaluate_round=mock_super_evaluate_round,
    )

    # Test fit
    client_manager = server.client_manager()
    with patch.object(FixedSamplingClientManager, "reset_sample", wraps=client_manager.reset_sample):
        fit_result = server.fit_round(test_server_round, None)
        client_manager.reset_sample.assert_called_once()  # type: ignore

    assert fit_result == (None, None, test_fit_results_and_failures)
    assert server.results_and_failures == test_fit_results_and_failures
    assert len(server.clients_metrics) == test_num_clients
    current_sample = server.client_manager().current_sample
    assert current_sample is not None
    for i in range(len(server.clients_metrics)):
        assert server.clients_metrics[i].cid == current_sample[i].cid
        assert server.clients_metrics[i].train_metrics == test_fit_results[i][1].metrics  # type: ignore

    # Test evaluate
    client_manager = server.client_manager()
    strategy = server.strategy
    with patch.object(FedDGGAServer, "calculate_weights_by_ga", wraps=server.calculate_weights_by_ga):
        with patch.object(FedDGGAServer, "apply_weights_to_results", wraps=server.apply_weights_to_results):
            with patch.object(FixedSamplingClientManager, "reset_sample", wraps=client_manager.reset_sample):
                with patch.object(FedAvg, "aggregate_fit", wraps=strategy.aggregate_fit):
                    evaluate_result = server.evaluate_round(test_server_round, None)
                    client_manager.reset_sample.assert_not_called()  # type: ignore
                    server.calculate_weights_by_ga.assert_called_once_with(test_server_round)  # type: ignore
                    server.apply_weights_to_results.assert_called_once()  # type: ignore
                    strategy.aggregate_fit.assert_called_once()  # type: ignore

    # adding the loss to the test evaluate result so we can make the assertions
    for result in test_evaluate_results:
        assert result[1] is not None
        result[1].metrics[test_fairness_metric] = test_loss

    assert evaluate_result == (test_loss, None, test_evaluate_results_and_failures)
    assert len(server.clients_metrics) == test_num_clients
    current_sample = server.client_manager().current_sample
    assert current_sample is not None
    for i in range(len(server.clients_metrics)):
        assert server.clients_metrics[i].cid == current_sample[i].cid
        assert server.clients_metrics[i].evaluation_metrics == test_evaluate_results[i][1].metrics  # type: ignore


def test_calculate_weights_by_ga() -> None:
    test_num_clients = 2
    test_total_rounds = 5
    test_server_round = 2

    server = _make_server(test_num_clients)
    server.num_rounds = test_total_rounds
    current_sample = server.client_manager().sample(test_num_clients)
    fairness_metric_name = server.fairness_metric.metric_name

    server.clients_metrics.append(
        ClientMetrics(
            cid=current_sample[0].cid,
            train_metrics={fairness_metric_name: 0.1},
            evaluation_metrics={fairness_metric_name: 0.2},
        )
    )
    server.clients_metrics.append(
        ClientMetrics(
            cid=current_sample[1].cid,
            train_metrics={fairness_metric_name: 0.5},
            evaluation_metrics={fairness_metric_name: 1.0},
        )
    )

    server.calculate_weights_by_ga(test_server_round)

    assert server.adjustment_weights == {
        current_sample[0].cid: approx(0.4708, abs=0.0005),
        current_sample[1].cid: approx(0.5291, abs=0.0005),
    }


def test_get_current_weight_step_size() -> None:
    test_total_rounds = 5
    test_server_round = 2
    test_num_clients = 2
    server = _make_server(test_num_clients)
    server.num_rounds = test_total_rounds

    step_size = server.get_current_weight_step_size(test_server_round)

    assert step_size == 0.16


def test_apply_weights_to_results() -> None:
    test_num_clients = 2
    server = _make_server(test_num_clients)
    current_sample = server.client_manager().sample(test_num_clients)
    server.adjustment_weights = {current_sample[0].cid: 0.4708, current_sample[1].cid: 0.5291}
    server.results_and_failures = (
        [
            [None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters([np.array([1.0, 2.0])]), 2, {})],  # type: ignore
            [None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters([np.array([3.0, 4.0])]), 2, {})],  # type: ignore
        ],
        [],
    )

    server.apply_weights_to_results()

    assert server.results_and_failures == (
        [
            [None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters([np.array([0.4708, 0.9416])]), 2, {})],
            [None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters([np.array([1.5873, 2.1164])]), 2, {})],
        ],
        [],
    )


def _make_server(num_clients: int) -> FedDGGAServer:
    server = FedDGGAServer()
    for i in range(num_clients):
        server.client_manager().register(CustomClientProxy(str(i)))
    server.client_manager().sample(num_clients)
    return server


def _setup_server_fit_and_evaluate_mocks(
    server: FedDGGAServer,
    loss: float,
    fit_results_and_failures: Tuple[List[Any], List[Any]],
    evaluate_results_and_failures: Tuple[List[Any], List[Any]],
    mock_fit_round: Mock,
    mock_evaluate_round: Mock,
) -> None:
    def fit_round_side_effect(
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        client_manager = server.client_manager()
        client_manager.sample(len(client_manager.clients))
        return None, None, fit_results_and_failures  # type: ignore

    mock_fit_round.side_effect = fit_round_side_effect

    def evaluate_round_side_effect(
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        return loss, None, evaluate_results_and_failures  # type: ignore

    mock_evaluate_round.side_effect = evaluate_round_side_effect
