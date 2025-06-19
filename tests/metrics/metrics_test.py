import numpy as np
import pytest
import torch

from fl4health.metrics import F1, Accuracy, BalancedAccuracy, BinarySoftDiceCoefficient, RocAuc


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


def test_binary_accuracy() -> None:
    accuracy_metric = Accuracy()

    pred1 = torch.Tensor([0, 1, 1, 0, 1])
    target1 = torch.Tensor([0, 0, 1, 1, 1])
    accuracy1 = accuracy_metric(pred1, target1)
    assert accuracy1 == 3.0 / 5.0


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


def test_roc_auc_metric() -> None:
    metric = RocAuc()

    logits1 = torch.Tensor(
        [
            [3.0, 1.0, 2.0],
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


def test_f1_metric() -> None:
    metric = F1()

    logits1 = torch.Tensor(
        [
            [3.0, 1.0, 2.0],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 1.1],
            [0.5, 3.0, 1.5],
        ]
    )
    target1 = torch.Tensor([0, 0, 2, 0, 2])
    assert metric(logits1, target1) == 0.68


def test_binary_soft_dice_coefficient_default_threshold() -> None:
    # Test with the default spatial and epsilon values
    metric = BinarySoftDiceCoefficient()
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    dice_of_one = metric(all_ones_logits, all_ones_targets)
    assert pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with random logits between 0 and 1
    np.random.seed(42)
    random_logits = torch.Tensor(np.random.rand(10, 1, 10, 10, 10))
    random_dice = metric(random_logits, all_ones_targets)
    assert pytest.approx(random_dice, abs=0.00001) == 0.6598031841976006

    # Test with intersection of zero to ensure edge case is equal to 0.0
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    dice_intersection_zero = metric(all_zeros_logits, all_ones_targets)
    assert pytest.approx(dice_intersection_zero, abs=0.000001) == 0.0

    # Test with union of zero to ensure edge case is equal to 1.0
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    all_zeros_target = torch.zeros((10, 1, 10, 10, 10))
    dice_union_zero = metric(all_zeros_logits, all_zeros_target)
    assert pytest.approx(dice_union_zero, abs=0.000001) == 1.0

    # Test with different spatial dimensions (i.e. a 2D target with two channels) and epsilon
    metric = BinarySoftDiceCoefficient(epsilon=0.1, spatial_dimensions=(2, 3))
    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero
    ones_and_zeros_logits[:, 1, :, :] = 0
    dice_coefficient = metric(ones_and_zeros_logits, all_ones_targets)
    # Union should be 100 and 50 for the two channels
    # Intersection should be 100 and 0 for the two channels
    # Dice should be (100)/(100 + 0.1) and 0 for the two channels
    # Mean over the 10 examples is equivalent to 0.5*(100)/(100 + 0.1)
    assert pytest.approx(dice_coefficient, abs=0.001) == 0.5 * (100) / (100 + 0.1)


def test_binary_soft_dice_coefficient_alt_threshold() -> None:
    # Change the threshold to 0.1
    metric = BinarySoftDiceCoefficient(logits_threshold=0.1)
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    dice_of_one = metric(all_ones_logits, all_ones_targets)
    assert pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with 0.25 in all entries, but with a lower threshold for classification as 1
    all_one_quarter_logits = 0.25 * torch.ones((10, 1, 10, 10, 10))
    dice_of_one = metric(all_one_quarter_logits, all_ones_targets)
    assert pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with a none threshold to ensure that the continuous dice coefficient is calculated
    metric = BinarySoftDiceCoefficient(logits_threshold=None)
    all_one_tenth_logits = 0.1 * torch.ones((10, 1, 10, 10, 10))
    continuous_dice = metric(all_one_tenth_logits, all_ones_targets)
    intersection = 100
    union = 0.5 * 1.1 * 1000
    dice_target = intersection / (union + 1e-7)
    assert pytest.approx(continuous_dice, abs=0.001) == dice_target


def test_metric_accumulation() -> None:
    a = Accuracy()

    pred1 = torch.eye(4)
    pred2 = torch.eye(4)
    pred3 = torch.eye(4)

    target1 = torch.arange(4)
    target2 = torch.arange(3, -1, -1)
    target3 = torch.tensor([0, 1, 1, 1])

    preds = [pred1, pred2, pred3]
    targets = [target1, target2, target3]

    for pred, target in zip(preds, targets):
        a.update(pred, target)

    assert a.compute()["accuracy"] == 0.5

    ba = BalancedAccuracy()

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

    for bl, bt in zip(batch_logits, batch_targets):
        ba.update(bl, bt)

    acc_m_balanced_accuracy = ba.compute()["balanced_accuracy"]

    # Accumulating the batches together results in recalls of (1.0, 1/5, 5/7) for 0, 1, 2 classes, these are then
    # averaged over the number of classes giving the correct balanced accuracy for the whole
    assert pytest.approx(acc_m_balanced_accuracy, abs=0.00001) == (1.0 + 1.0 / 5.0 + 5.0 / 7.0) / 3.0
