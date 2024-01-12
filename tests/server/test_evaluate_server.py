import datetime
from unittest.mock import Mock, patch

from freezegun import freeze_time

from fl4health.client_managers.base_sampling_manager import SimpleClientManager
from fl4health.server.evaluate_server import EvaluateServer


@patch("fl4health.server.evaluate_server.EvaluateServer.federated_evaluate")
@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit(mock_federated_evaluate: Mock) -> None:
    test_evaluate_metrics = {"test evaluate metrics": 123}
    mock_federated_evaluate.return_value = (None, test_evaluate_metrics, None)

    evaluate_server = EvaluateServer(SimpleClientManager(), 0.5)
    evaluate_server.fit(3, None)

    assert evaluate_server.metrics_reporter.metrics == {
        "type": "server",
        "fit_start": datetime.datetime(2012, 12, 12, 12, 12, 12),
        "fit_end": datetime.datetime(2012, 12, 12, 12, 12, 12),
        "metrics": test_evaluate_metrics,
    }
