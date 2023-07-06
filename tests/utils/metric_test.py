import pytest
import torch

from fl4health.utils.metrics import F1, ROC_AUC, AccumulationMeter, Accuracy, AverageMeter, BalancedAccuracy


def test_accuracy_metric() -> None:
    accuracy_metric = Accuracy()

    pred1 = torch.eye(5)
    target1 = torch.arange(0, 5)
    accuracy1 = accuracy_metric(pred1, target1)
    assert accuracy1 == 1.0

    pred2 = torch.eye(4)
    target2 = torch.tensor([0, 1, 1, 3])
    accuracy2 = accuracy_metric(pred2, target2)
    assert accuracy2 == 0.75


def test_average_meter() -> None:
    am = AverageMeter([Accuracy()])

    pred1 = torch.eye(4)
    pred2 = torch.eye(4)
    pred3 = torch.eye(4)

    target1 = torch.arange(4)
    target2 = torch.arange(3, -1, -1)
    target3 = torch.tensor([0, 1, 1, 1])

    preds = [pred1, pred2, pred3]
    targets = [target1, target2, target3]

    for pred, target in zip(preds, targets):
        am.update(pred, target)

    assert am.compute()["accuracy"] == 0.5

    am2 = AverageMeter([Accuracy("global_accuracy"), Accuracy("local_accuracy")])

    for pred, target in zip(preds, targets):
        am2.update(pred, target)

    assert am2.compute()["global_accuracy"] == 0.5
    assert am2.compute()["local_accuracy"] == 0.5


def test_balanced_accuracy() -> None:
    metric = BalancedAccuracy()

    logits = torch.Tensor([[0.75, 0.25], [0.12, 0.88], [0.9, 0.1], [0.94, 0.06], [0.78, 0.22], [0.08, 0.92]])
    target = torch.Tensor([0, 1, 0, 0, 1, 0])

    assert metric(logits, target) == 0.625

    logits = torch.Tensor(
        [
            [0.75, 0.20, 0.05],
            [0.88, 0.06, 0.06],
            [0.1, 0.1, 0.8],
            [0.94, 0.03, 0.03],
            [0.11, 0.22, 0.67],
            [0.02, 0.92, 0.06],
        ]
    )
    target = torch.Tensor([0, 1, 2, 0, 1, 2])

    assert metric(logits, target) == 0.5


def test_accumulation_meter() -> None:
    accumulation_meter = AccumulationMeter([BalancedAccuracy()])
    average_meter = AverageMeter([BalancedAccuracy()])

    logits1 = torch.Tensor(
        [
            [0.75, 0.20, 0.05],
            [0.88, 0.06, 0.06],
            [0.1, 0.1, 0.8],
            [0.94, 0.03, 0.03],
            [0.11, 0.22, 0.67],
            [0.02, 0.92, 0.06],
        ]
    )
    logits2 = torch.Tensor(
        [
            [0.75, 0.20, 0.05],
            [0.88, 0.06, 0.06],
            [0.1, 0.1, 0.8],
            [0.94, 0.03, 0.03],
            [0.11, 0.22, 0.67],
            [0.02, 0.92, 0.06],
            [0.11, 0.22, 0.67],
            [0.02, 0.06, 0.92],
        ]
    )
    logits3 = torch.Tensor(
        [
            [0.75, 0.20, 0.05],
            [0.08, 0.86, 0.06],
            [0.1, 0.1, 0.8],
        ]
    )
    target1 = torch.Tensor([0, 1, 2, 0, 1, 2])
    target2 = torch.Tensor([0, 1, 2, 0, 1, 2, 2, 2])
    target3 = torch.Tensor([0, 1, 2])

    batch_logits = [logits1, logits2, logits3]
    batch_targets = [target1, target2, target3]

    for logits, targets in zip(batch_logits, batch_targets):
        accumulation_meter.update(logits, targets)
        average_meter.update(logits, targets)

    avg_m_balanced_accuracy = average_meter.compute()["balanced_accuracy"]
    acc_m_balanced_accuracy = accumulation_meter.compute()["balanced_accuracy"]

    # Balanced accuracy for each batch is (0.5, 1.0, (1.75/3.0) respectively
    # Batch sizes are (6, 3, 8), respectively.
    assert pytest.approx(avg_m_balanced_accuracy, abs=0.00001) == (
        0.5 * (6.0 / 17) + 1.0 * (3.0 / 17) + (1.75 / 3.0) * ((8.0 / 17))
    )
    # Accumulating the batches together results in recalls of (1.0, 1/5, 5/7) for 0, 1, 2 classes, these are then
    # averaged over the number of classes giving the correct balanced accuracy for the whole
    assert pytest.approx(acc_m_balanced_accuracy, abs=0.00001) == (1.0 + 1.0 / 5.0 + 5.0 / 7.0) / 3.0


def test_ROC_AUC_metric() -> None:
    metric = ROC_AUC()

    logits1 = torch.Tensor(
        [
            [3, 1, 2],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 0.1],
            [3, 10, 2],
            [1.5, 0.5, 0.5],
        ]
    )
    target1 = torch.Tensor([0, 1, 2, 0, 1, 2])

    assert metric(logits1, target1) == 0.75

    logits2 = torch.Tensor(
        [
            [0.75, 0.20, 0.05],
            [0.08, 0.86, 0.06],
            [0.1, 0.1, 0.8],
        ]
    )
    target2 = torch.Tensor([0, 0, 2])
    # It should raise ValueError since AUC ROC score is not defined in this case.
    with pytest.raises(ValueError):
        metric(logits2, target2)


def test_F1_metric() -> None:
    metric = F1()

    logits1 = torch.Tensor(
        [
            [3, 1, 2],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 1.1],
            [0.5, 3.0, 1.5],
        ]
    )
    target1 = torch.Tensor([0, 0, 2, 0, 2])

    assert metric(logits1, target1) == 0.68
