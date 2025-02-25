import pytest
import torch

from fl4health.utils.metrics import F1, Accuracy, MetricManager


def test_metric_manager() -> None:
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

    mm = MetricManager([F1(), Accuracy()], "test")

    for logits, target in zip(logits_list, target_list):
        preds = {"prediction": logits}
        mm.update(preds, target)
    metrics = mm.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.80285714285, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.8


def test_metric_manager_clear() -> None:
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

    mm = MetricManager([F1(), Accuracy()], "test")

    for logits, target in zip(logits_list, target_list):
        preds = {"prediction": logits}
        mm.update(preds, target)
    metrics = mm.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.80285714285, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.8

    mm.clear()

    preds = {"prediction": logits1}
    mm.update(preds, target1)
    metrics = mm.compute()

    assert metrics["test - prediction - F1 score"] == pytest.approx(0.68, abs=0.00001)
    assert metrics["test - prediction - accuracy"] == 0.6
