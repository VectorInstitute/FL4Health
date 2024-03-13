from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

from flwr.common.typing import Code, EvaluateRes, FitRes, Parameters, Scalar, Status
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.strategy.fedavg import FedAvg

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.server.feddg_ga_server import FedDGGAServer
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
    test_fit_results = [
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, {"test_fit_metric": 1.1, "val - loss": 1.2})],
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, {"test_fit_metric": 2.1, "val - loss": 2.2})],
    ]
    test_evaluate_results = [
        [None, EvaluateRes(Status(Code.OK, ""), 3.0, 2, {"test_eval_metric": 4.0})],
        [None, EvaluateRes(Status(Code.OK, ""), 5.0, 2, {"test_eval_metric": 6.0})],
    ]
    test_failures: List[Any] = []
    test_fit_results_and_failures = (test_fit_results, test_failures)
    test_evaluate_results_and_failures = (test_evaluate_results, test_failures)
    server = _setup_server(
        num_clients=test_num_clients,
        loss=test_loss,
        fit_results_and_failures=deepcopy(test_fit_results_and_failures),
        evaluate_results_and_failures=deepcopy(test_evaluate_results_and_failures),
        mock_fit_round=mock_super_fit_round,
        mock_evaluate_round=mock_super_evaluate_round,
    )
    server.num_rounds = test_total_rounds

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
        result[1].metrics["val - loss"] = test_loss

    assert evaluate_result == (test_loss, None, test_evaluate_results_and_failures)
    assert len(server.clients_metrics) == test_num_clients
    current_sample = server.client_manager().current_sample
    assert current_sample is not None
    for i in range(len(server.clients_metrics)):
        assert server.clients_metrics[i].cid == current_sample[i].cid
        assert server.clients_metrics[i].evaluation_metrics == test_evaluate_results[i][1].metrics  # type: ignore


def _setup_server(
    num_clients: int,
    loss: float,
    fit_results_and_failures: Tuple[List[Any], List[Any]],
    evaluate_results_and_failures: Tuple[List[Any], List[Any]],
    mock_fit_round: Mock,
    mock_evaluate_round: Mock,
) -> FedDGGAServer:
    server = FedDGGAServer()
    for i in range(num_clients):
        server.client_manager().register(CustomClientProxy(str(i)))

    def fit_round_side_effect(
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        server.client_manager().sample(num_clients)
        return None, None, fit_results_and_failures  # type: ignore

    mock_fit_round.side_effect = fit_round_side_effect

    def evaluate_round_side_effect(
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        server.client_manager().sample(num_clients)
        return loss, None, evaluate_results_and_failures  # type: ignore

    mock_evaluate_round.side_effect = evaluate_round_side_effect

    return server
