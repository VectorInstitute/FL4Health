import math
from typing import Dict

import pytest
import torch
import torch.nn as nn
from flwr.common.typing import Scalar

from fl4health.clients.evaluate_client import EvaluateClient
from tests.clients.fixtures import get_evaluation_client  # noqa


class SingleLayer(nn.Module):
    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(100, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_evaluate_merge_metrics(caplog: pytest.LogCaptureFixture) -> None:
    global_metrics: Dict[str, Scalar] = {"global_metric_1": 0.22, "local_metric_2": 0.11}
    local_metrics: Dict[str, Scalar] = {"local_metric_1": 0.1, "local_metric_2": 0.99}
    merged_metrics = EvaluateClient.merge_metrics(global_metrics, local_metrics)
    # Test merge is good, local metrics are folded in last, so they take precedence when overlap exists
    assert merged_metrics == {"global_metric_1": 0.22, "local_metric_1": 0.1, "local_metric_2": 0.99}
    # Test that we are warned about duplicate metric keys
    assert "metric_name: local_metric_2 already exists in dictionary." in caplog.text

    # test whether merge works alright given missing dictionaries.
    merged_metrics = EvaluateClient.merge_metrics(None, local_metrics)
    assert merged_metrics == local_metrics
    merged_metrics = EvaluateClient.merge_metrics(global_metrics, None)
    assert merged_metrics == global_metrics


@pytest.mark.parametrize("model", [SingleLayer()])
def test_evaluating_identical_global_and_local_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_meter_accuracy"], abs=0.0001) == 0.0
    assert pytest.approx(metrics["global_eval_meter_accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayer()])
def test_evaluating_different_global_and_local_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client
    evaluate_client.global_model = SingleLayer(seed=37)

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.5104386806
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_meter_accuracy"], abs=0.0001) == 0.0
    assert pytest.approx(metrics["global_eval_meter_accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayer()])
def test_evaluating_only_local_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client
    evaluate_client.global_model = None

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert "global_loss_checkpoint" not in metrics
    assert pytest.approx(metrics["local_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["local_eval_meter_accuracy"], abs=0.0001) == 0.0


@pytest.mark.parametrize("model", [SingleLayer()])
def test_evaluating_only_global_models(get_evaluation_client: EvaluateClient) -> None:  # noqa
    evaluate_client = get_evaluation_client
    evaluate_client.local_model = None

    loss, metrics = evaluate_client.validate()
    assert math.isnan(loss)
    assert "local_loss_checkpoint" not in metrics
    assert pytest.approx(metrics["global_loss_checkpoint"], abs=0.0001) == 1.43826544285
    assert pytest.approx(metrics["global_eval_meter_accuracy"], abs=0.0001) == 0.0
