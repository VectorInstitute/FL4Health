import torch
from pytest import approx

from fl4health.metrics.compound_metrics import TransformsMetric
from tests.metrics.metric_utility import AccuracyForTest


def test_none_transform_metric_computation() -> None:
    preds_1 = torch.Tensor([0.9, 0.7, 0.6, 0.9, 0.1, 0.2, 0.99])
    targets_1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_2 = torch.Tensor([0.19, 0.75, 0.26, 0.49, 0.12, 0.92, 0.99])
    targets_2 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_3 = torch.Tensor([0.19, 0.75, 0.26, 0.49])
    targets_3 = torch.Tensor([1.0, 0.0, 1.0, 0.0])

    accuracy_metric = AccuracyForTest("accuracy")
    transform_accuracy_metric = TransformsMetric(accuracy_metric, None, None)

    # First accumulate two batches of updates before computation
    transform_accuracy_metric.update(preds_1, targets_1)
    transform_accuracy_metric.update(preds_2, targets_2)

    # Compute metric. Should be identical to accuracy.
    metrics_1 = transform_accuracy_metric.compute()
    assert metrics_1["accuracy"] == approx(6.0 / 14.0)

    # Now we clear the metric, as we are ready to start a new score
    transform_accuracy_metric.clear()

    transform_accuracy_metric.update(preds_3, targets_3)

    # Compute metric. Should be identical to accuracy.
    metrics_2 = transform_accuracy_metric.compute()
    assert metrics_2["accuracy"] == approx(1.0 / 4.0)


def test_identity_transforms_metric_computation() -> None:
    preds_1 = torch.Tensor([0.9, 0.7, 0.6, 0.9, 0.1, 0.2, 0.99])
    targets_1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_2 = torch.Tensor([0.19, 0.75, 0.26, 0.49, 0.12, 0.92, 0.99])
    targets_2 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_3 = torch.Tensor([0.19, 0.75, 0.26, 0.49])
    targets_3 = torch.Tensor([1.0, 0.0, 1.0, 0.0])

    accuracy_metric = AccuracyForTest("accuracy")
    transform_accuracy_metric = TransformsMetric(
        accuracy_metric, [lambda x: x, lambda x: x], [lambda x: x, lambda x: x]
    )

    # First accumulate two batches of updates before computation
    transform_accuracy_metric.update(preds_1, targets_1)
    transform_accuracy_metric.update(preds_2, targets_2)

    # Compute metric. Should be identical to accuracy.
    metrics_1 = transform_accuracy_metric.compute()
    assert metrics_1["accuracy"] == approx(6.0 / 14.0)

    # Now we clear the metric, as we are ready to start a new score
    transform_accuracy_metric.clear()

    transform_accuracy_metric.update(preds_3, targets_3)

    # Compute metric. Should be identical to accuracy.
    metrics_2 = transform_accuracy_metric.compute()
    assert metrics_2["accuracy"] == approx(1.0 / 4.0)


def binarize(x: torch.Tensor) -> torch.Tensor:
    mask_1 = x <= 1.5
    mask_2 = x > 1.5
    x[mask_1] = 0.0
    x[mask_2] = 1.0
    return x


def test_multiple_transforms_metric_computation() -> None:
    preds_1 = torch.Tensor([0.9, 0.7, 0.6, 0.9, 0.1, 0.2, 0.99])
    targets_1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_2 = torch.Tensor([0.19, 0.75, 0.26, 0.49, 0.12, 0.92, 0.99])
    targets_2 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_3 = torch.Tensor([0.19, 0.75, 0.26, 0.49])
    targets_3 = torch.Tensor([1.0, 0.0, 1.0, 0.0])

    accuracy_metric = AccuracyForTest("accuracy")
    transform_accuracy_metric = TransformsMetric(
        accuracy_metric,
        [lambda x: x + 1.0, lambda x: binarize(x)],
        [lambda x: x + 1.0, lambda x: x - 1.0],
    )

    # First accumulate two batches of updates before computation
    transform_accuracy_metric.update(preds_1, targets_1)
    transform_accuracy_metric.update(preds_2, targets_2)

    # Compute metric. Should be the same as accuracy.
    metrics_1 = transform_accuracy_metric.compute()
    assert metrics_1["accuracy"] == approx(6.0 / 14.0)

    # Now we clear the metric, as we are ready to start a new score
    transform_accuracy_metric.clear()

    transform_accuracy_metric.update(preds_3, targets_3)

    # Compute metric. Should be the same as accuracy.
    metrics_2 = transform_accuracy_metric.compute()
    assert metrics_2["accuracy"] == approx(1.0 / 4.0)


def test_transforms_changing_metric_computation() -> None:
    preds_1 = torch.Tensor([0.9, 0.7, 0.6, 0.9, 0.1, 0.2, 0.99])
    targets_1 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_2 = torch.Tensor([0.19, 0.75, 0.26, 0.49, 0.12, 0.92, 0.99])
    targets_2 = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    preds_3 = torch.Tensor([0.19, 0.75, 0.26, 0.49])
    targets_3 = torch.Tensor([1.0, 0.0, 1.0, 0.0])

    accuracy_metric = AccuracyForTest("accuracy")
    transform_accuracy_metric = TransformsMetric(
        accuracy_metric,
        [lambda x: x + 1.0],
        [lambda x: x + 1.0, lambda x: x - 1.0],
    )

    # First accumulate two batches of updates before computation
    transform_accuracy_metric.update(preds_1, targets_1)
    transform_accuracy_metric.update(preds_2, targets_2)

    # Should be different than just plain accuracy, as all predictions are bumped to essentially be 1.0
    metrics_1 = transform_accuracy_metric.compute()
    assert metrics_1["accuracy"] == approx(8.0 / 14.0)

    # Now we clear the metric, as we are ready to start a new score
    transform_accuracy_metric.clear()

    transform_accuracy_metric.update(preds_3, targets_3)

    # Should be different than just plain accuracy, as all predictions are bumped to essentially be 1.0
    metrics_2 = transform_accuracy_metric.compute()
    assert metrics_2["accuracy"] == approx(2.0 / 4.0)
