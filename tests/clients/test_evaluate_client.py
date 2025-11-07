import datetime
import math
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from flwr.common.typing import Scalar
from freezegun import freeze_time

from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from tests.clients.fixtures import get_basic_client, get_evaluation_client  # noqa
from tests.test_utils.assert_metrics_dict import assert_metrics_dict
from tests.test_utils.models_for_test import SingleLayerWithSeed


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_evaluate_merge_metrics(caplog: pytest.LogCaptureFixture) -> None:
    global_metrics: dict[str, Scalar] = {
        "global_metric_1": 0.22,
        "local_metric_2": 0.11,
    }
    local_metrics: dict[str, Scalar] = {"local_metric_1": 0.1, "local_metric_2": 0.99}
    merged_metrics = EvaluateClient.merge_metrics(global_metrics, local_metrics)
    # Test merge is good, local metrics are folded in last, so they take precedence when overlap exists
    assert merged_metrics == {
        "global_metric_1": 0.22,
        "local_metric_1": 0.1,
        "local_metric_2": 0.99,
    }
    # Test that we are warned about duplicate metric keys
    assert "metric_name: local_metric_2 already exists in dictionary." in caplog.text

    # test whether merge works alright given missing dictionaries.
    merged_metrics = EvaluateClient.merge_metrics(None, local_metrics)
    assert merged_metrics == local_metrics
    merged_metrics = EvaluateClient.merge_metrics(global_metrics, None)
    assert merged_metrics == global_metrics


@pytest.mark.parametrize("model", [SingleLayerWithSeed()])
def test_evaluating_identical_global_and_local_models(
    get_evaluation_client: EvaluateClient,  # noqa
) -> None:
    evaluate_client = get_evaluation_client

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0
    assert pytest.approx(metrics["global_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayerWithSeed()])
def test_evaluating_different_global_and_local_models(
    get_evaluation_client: EvaluateClient,  # noqa
) -> None:
    evaluate_client = get_evaluation_client
    evaluate_client.global_model = SingleLayerWithSeed(seed=37)

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.5104386806
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0
    assert pytest.approx(metrics["global_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayerWithSeed()])
def test_evaluating_only_local_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client
    evaluate_client.global_model = None

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert "global_loss_checkpoint" not in metrics
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayerWithSeed()])
def test_evaluating_only_global_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client
    evaluate_client.local_model = None

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert "local_loss_checkpoint" not in metrics
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["global_eval_manager - prediction - accuracy"], abs=0.0001) == 0.0


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_setup_client() -> None:
    reporter = JsonReporter()
    evaluate_client = MockEvaluateClient(reporters=[reporter])
    evaluate_client.setup_client({})

    metrics_to_assert = {
        "host_type": "client",
        "initialized": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
    }
    errors = assert_metrics_dict(metrics_to_assert, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_evaluate() -> None:
    test_loss = 123.123
    test_metrics: dict[str, bool | bytes | float | int | str] = {"test_metric": 1234}

    reporter = JsonReporter()
    evaluate_client = MockEvaluateClient(loss=test_loss, metrics=test_metrics, reporters=[reporter])
    evaluate_client.evaluate([], {})
    metric_dict = {
        "host_type": "client",
        "initialized": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "rounds": {
            0: {
                "eval_metrics": test_metrics,
                "eval_loss": test_loss,
                "eval_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
            }
        },
    }
    errors = assert_metrics_dict(metric_dict, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"


class MockEvaluateClient(EvaluateClient):
    def __init__(
        self,
        loss: float | None = None,
        metrics: dict[str, Scalar] | None = None,
        reporters: Sequence[BaseReporter] | None = None,
    ):
        super().__init__(Path(""), [], DEVICE, reporters=reporters)

        # Mocking methods
        self.get_data_loader = MagicMock()  # type: ignore
        mock_data_loader = MagicMock()  # type: ignore
        mock_data_loader.dataset = []
        self.get_data_loader.return_value = (mock_data_loader,)
        self.get_criterion = MagicMock()  # type: ignore
        self.get_local_model = MagicMock()  # type: ignore
        self.validate = MagicMock()  # type: ignore
        self.validate.return_value = loss, metrics
