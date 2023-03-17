import torch

from fl4health.utils.metrics import Accuracy, AverageMeter


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

    assert am.compute()["_accuracy"] == 0.5
