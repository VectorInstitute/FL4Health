import datetime
from unittest.mock import Mock, patch

from freezegun import freeze_time

from fl4health.client_managers.base_sampling_manager import SimpleClientManager
from fl4health.reporting import JsonReporter
from fl4health.servers.evaluate_server import EvaluateServer
from tests.test_utils.assert_metrics_dict import assert_metrics_dict


@patch("fl4health.servers.evaluate_server.EvaluateServer.federated_evaluate")
@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit(mock_federated_evaluate: Mock) -> None:
    pass
    test_evaluate_metrics = {"test evaluate metrics": 123}
    mock_federated_evaluate.return_value = (None, test_evaluate_metrics, None)

    reporter = JsonReporter()
    evaluate_server = EvaluateServer(SimpleClientManager(), 0.5, reporters=[reporter])
    evaluate_server.fit(3, None)
    metrics_to_assert = {
        "host_type": "server",
        "fit_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "fit_end": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "fit_metrics": test_evaluate_metrics,
    }
    errors = assert_metrics_dict(metrics_to_assert, reporter.metrics)
    assert len(errors) == 0, f"Metric check failed. Errors: {errors}"
