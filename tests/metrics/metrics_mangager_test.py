import pytest
import torch

from fl4health.metrics import F1, Accuracy
from fl4health.metrics.metric_managers import MetricManager


def get_logits_and_targets() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    logits1 = torch.Tensor(
        [
            [0.8, 0.05, 0.15],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 0.6],
            [0.4, 0.1, 0.5],
            [0.1, 0.6, 0.3],
        ]
    )
    target1 = torch.Tensor([0, 0, 2, 0, 2])

    logits2 = torch.Tensor(
        [
            [0.4, 0.5, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
            [0.75, 0.15, 0.1],
            [0.1, 0.6, 0.3],
        ]
    )
    target2 = torch.Tensor([1, 2, 2, 0, 1])

    logits_list = [logits1, logits2]
    target_list = [target1, target2]

    return logits_list, target_list


LOGITS, TARGETS = get_logits_and_targets()


def test_metric_manager() -> None:
    metric_manager = MetricManager([F1(), Accuracy()], "test")

    for logits, target in zip(LOGITS, TARGETS):
        preds = {"prediction": logits}
        metric_manager.update(preds, target)
    metrics = metric_manager.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.80285714285, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.8


def test_metric_manager_clear() -> None:
    metric_manager = MetricManager([F1(), Accuracy()], "test")

    for logits, target in zip(LOGITS, TARGETS):
        preds = {"prediction": logits}
        metric_manager.update(preds, target)
    metrics = metric_manager.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.80285714285, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.8

    metric_manager.clear()

    preds = {"prediction": LOGITS[0]}
    metric_manager.update(preds, TARGETS[0])
    metrics = metric_manager.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.68, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.6
