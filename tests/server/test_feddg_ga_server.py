from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

from flwr.common.typing import Code, FitRes, Parameters, Scalar, Status
from flwr.server.server import FitResultsAndFailures

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


@patch("fl4health.server.feddg_ga_server.FlServer.fit_round")
def test_fit_round(mock_super_fit_round: Mock) -> None:
    server = FedDGGAServer()
    test_num_clients = 2
    for i in range(test_num_clients):
        server.client_manager().register(CustomClientProxy(str(i)))

    test_metrics: List[Dict[str, Union[bool, bytes, float, int, str]]] = [
        {"test_metric": 123},
        {"test_metric": 456},
    ]
    test_results = [
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, test_metrics[0])],
        [None, FitRes(Status(Code.OK, ""), Parameters([], ""), 2, test_metrics[1])],
    ]
    test_failures = []  # type: ignore
    test_server_round = 2

    def fit_round_side_effect(
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        server.client_manager().sample(test_num_clients)
        return None, None, (test_results, test_failures)  # type: ignore

    mock_super_fit_round.side_effect = fit_round_side_effect

    test_fit_result = server.fit_round(test_server_round, None)

    assert test_fit_result == (None, None, (test_results, test_failures))
    assert server.results_and_failures == (test_results, test_failures)
    assert len(server.clients_metrics) == test_num_clients
    current_sample = server.client_manager().current_sample
    assert current_sample is not None
    for i in range(len(server.clients_metrics)):
        assert server.clients_metrics[i].cid == current_sample[i].cid
        assert server.clients_metrics[i].train_metrics == test_metrics[i]

    # TODO: assert reset sampling called
