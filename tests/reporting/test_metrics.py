import datetime
import json
import os
from unittest.mock import Mock, patch

from fl4health.reporting.metrics import MetricsReporter


@patch("fl4health.reporting.metrics.uuid")
def test_metrics_reporter_init(mock_uuid: Mock) -> None:
    test_uuid = "test uuid"
    mock_uuid.uuid4.return_value = test_uuid

    metrics_reporter = MetricsReporter()

    assert metrics_reporter.run_id == test_uuid
    assert metrics_reporter.metrics == {}


def test_metrics_reporter_add_to_metrics() -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}

    metrics_reporter = MetricsReporter()

    metrics_reporter.add_to_metrics(test_data_1)
    assert metrics_reporter.metrics == test_data_1

    metrics_reporter.add_to_metrics(test_data_2)
    assert metrics_reporter.metrics == {**test_data_1, **test_data_2}


def test_metrics_reporter_add_to_metrics_at_round() -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}

    metrics_reporter = MetricsReporter()

    metrics_reporter.add_to_metrics_at_round(2, test_data_1)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: test_data_1,
        },
    }

    metrics_reporter.add_to_metrics_at_round(4, test_data_1)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: test_data_1,
            4: test_data_1,
        },
    }

    metrics_reporter.add_to_metrics_at_round(2, test_data_2)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: {**test_data_1, **test_data_2},
            4: test_data_1,
        },
    }


def test_metrics_reporter_dump() -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}
    test_date = datetime.datetime.now()
    test_folder = "tests/reporting"
    test_run_id = "test"
    test_json_file_name = f"{test_folder}/{test_run_id}.json"

    metrics_reporter = MetricsReporter(run_id=test_run_id, output_folder=test_folder)
    metrics_reporter.add_to_metrics(test_data_1)
    metrics_reporter.add_to_metrics({"date": test_date})
    metrics_reporter.add_to_metrics_at_round(2, test_data_2)
    metrics_reporter.dump()

    with open(test_json_file_name, "r") as file:
        json_data = json.load(file)

    assert json_data == {
        **test_data_1,  # type: ignore
        "date": str(test_date),
        "rounds": {
            "2": test_data_2,
        },
    }

    os.remove(test_json_file_name)
