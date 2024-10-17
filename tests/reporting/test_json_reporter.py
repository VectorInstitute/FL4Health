import datetime
import json
from pathlib import Path
from unittest.mock import Mock, patch

from fl4health.reporting import JsonReporter


@patch("fl4health.reporting.json_reporter.uuid")
def test_json_reporter_init(mock_uuid: Mock) -> None:
    test_uuid = "test uuid"
    mock_uuid.uuid4.return_value = test_uuid

    metrics_reporter = JsonReporter()
    metrics_reporter.initialize()

    assert metrics_reporter.run_id == test_uuid
    assert metrics_reporter.metrics == {}


def test_json_reporter_add_summary_data() -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}

    metrics_reporter = JsonReporter()

    metrics_reporter.report(test_data_1)
    assert metrics_reporter.metrics == test_data_1

    metrics_reporter.report(test_data_2)
    assert metrics_reporter.metrics == {**test_data_1, **test_data_2}


def test_metrics_reporter_add_round_data() -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}

    metrics_reporter = JsonReporter()

    metrics_reporter.report(test_data_1, round=2)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: test_data_1,
        },
    }

    metrics_reporter.report(test_data_1, round=4)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: test_data_1,
            4: test_data_1,
        },
    }

    metrics_reporter.report(test_data_2, round=2)
    assert metrics_reporter.metrics == {
        "rounds": {
            2: {**test_data_1, **test_data_2},
            4: test_data_1,
        },
    }


def test_metrics_reporter_dump(tmp_path: Path) -> None:
    test_data_1 = {"test data 1": 123}
    test_data_2 = {"test data 2": 456}
    test_date = str(datetime.datetime.now())
    test_run_id = "test"
    test_json_file_name = f"{tmp_path}/{test_run_id}.json"

    metrics_reporter = JsonReporter(run_id=test_run_id, output_folder=tmp_path)
    metrics_reporter.report(test_data_1)
    metrics_reporter.report({"date": test_date})
    metrics_reporter.report(test_data_2, round=2)
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
