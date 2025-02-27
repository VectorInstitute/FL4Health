import math

import numpy as np
import pytest
import torch
from pytest import approx

from fl4health.utils.metrics import (
    F1,
    ROC_AUC,
    Accuracy,
    BalancedAccuracy,
    HardDICE,
    MetricManager,
    SoftDICE,
)


def test_hard_dice_metric_1d_and_clear() -> None:
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=None, ignore_null=True)

    # Test with two 1D examples
    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)

    hd.clear()  # Clear to restart
    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.5)


def test_hard_dice_metric_1d_accumulation() -> None:
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=None, ignore_null=True)

    # Test accumulation
    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.5)

    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.65)  # Avg of 0.8 and 0.5


def test_hard_dice_metric_2d() -> None:
    # Test higher dimension examples. Shape is (1, 2, 4)
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=None, ignore_null=True)
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)


def test_hard_dice_metric_ignore_null_true() -> None:
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=None, ignore_null=True)

    # Initial test
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)

    # Test that ignore null is working by adding a null sample
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)

    # Test that NaN is returned when all dice coefficients are null and ignore_null is True. Should return NaN
    hd.clear()
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert isinstance(result["DICE"], float)  # Appeases mypy, ensure its a float before checking if its NaN
    assert math.isnan(result["DICE"])


def test_hard_dice_metric_ignore_null_false() -> None:
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=None, ignore_null=False)

    # Initial test
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)

    # Test that ignore null is working by adding a null sample. Should have dice score set to 1
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.9)

    # Test that NaN is returned when all dice coefficients are null and ignore_null is True. Should return NaN
    hd.clear()
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(1)


def test_hard_dice_metric_ignore_background() -> None:
    # Test ignore background
    hd = HardDICE(name="DICE", along_axes=(0,), ignore_background_axis=1, ignore_null=True)
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(1)

    # Test that accumulation works still
    p = torch.tensor([[[0, 0, 0, 1], [0, 0, 0, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [0, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.75)


def test_hard_dice_metric_along_axes() -> None:
    # Test computing dice coefficients along different axes. Shape (n, 2, 4)
    hd = HardDICE(name="DICE", along_axes=(1,), ignore_background_axis=None, ignore_null=True)
    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.75)  # (0.5 + 1) / 2

    # Ensure first dimension is being summed over by adding another sample
    p = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(11 / 24)  # (2/8 + 2/3) / 2

    # Test along last dimension
    hd = HardDICE(name="DICE", along_axes=(2,), ignore_background_axis=None, ignore_null=True)
    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(5 / 6)  # (1 + 2/3 + 2/3 + 1) / 4

    # Test using along multiple axes. Shape (3, 2, 4)
    hd = HardDICE(name="DICE", along_axes=(0, 1), ignore_background_axis=None, ignore_null=True)
    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    p = torch.tensor([[[1, 1, 1, 1], [0, 0, 0, 1]]])
    t = torch.tensor([[[1, 1, 1, 1], [0, 1, 1, 1]]])
    hd.update(p, t)
    p = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 1, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(4 / 6)  # (0.5 + 1 + 1 + 0.5 + 0 + 1) / 6


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


def test_ROC_AUC_metric() -> None:
    metric = ROC_AUC()

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


def test_F1_metric() -> None:
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


def test_hard_dice_default_threshold() -> None:
    # Test with the default spatial and epsilon values
    metric = HardDICE(name="DICE", along_axes=(0, 1), ignore_null=False, binarize=0.5)
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    metric.update(all_ones_logits, all_ones_targets)
    dice_of_one = metric.compute()["DICE"]
    pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with random logits between 0 and 1
    metric.clear()
    np.random.seed(42)
    random_logits = torch.Tensor(np.random.rand(10, 1, 10, 10, 10))
    metric.update(random_logits, all_ones_targets)
    random_dice = metric.compute()["DICE"]
    pytest.approx(random_dice, abs=0.00001) == 0.6598031841976006

    # Test with intersection of zero to ensure edge case is equal to 0.0
    metric.clear()
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    metric.update(all_zeros_logits, all_ones_targets)
    dice_intersection_zero = metric.compute()["DICE"]
    pytest.approx(dice_intersection_zero, abs=0.000001) == 0.0

    # Test with union of zero to ensure edge case is equal to 1.0
    metric.clear()
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    all_zeros_target = torch.zeros((10, 1, 10, 10, 10))
    metric.update(all_zeros_logits, all_zeros_target)
    dice_union_zero = metric.compute()["DICE"]
    pytest.approx(dice_union_zero, abs=0.000001) == 1.0

    # Test with different spatial dimensions (i.e. a 2D target with two channels) and epsilon
    metric.clear()
    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero
    ones_and_zeros_logits[:, 1, :, :] = 0
    metric.update(ones_and_zeros_logits, all_ones_targets)
    dice_coefficient = metric.compute()["DICE"]
    # Union should be 100 and 50 for the two channels
    # Intersection should be 100 and 0 for the two channels
    # Dice should be (100)/(100 + 0.1) and 0 for the two channels
    # Mean over the 10 examples is equivalent to 0.5*(100)/(100 + 0.1)
    pytest.approx(dice_coefficient, abs=0.001) == 0.5 * (100) / (100 + 0.1)


def test_hard_and_soft_dice_alt_threshold() -> None:
    # Change the threshold to 0.1
    metric = HardDICE(name="DICE", along_axes=(0, 1), ignore_null=False, binarize=0.1)

    # Test all TP's
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    metric.update(all_ones_logits, all_ones_targets)
    dice_of_one = metric.compute()["DICE"]
    pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with 0.25 in all entries, but with a lower threshold for classification as 1
    metric.clear()
    all_one_quarter_logits = 0.25 * torch.ones((10, 1, 10, 10, 10))
    metric.update(all_one_quarter_logits, all_ones_logits)
    dice_of_one = metric.compute()["DICE"]
    pytest.approx(dice_of_one, abs=0.001) == 1.0

    # Test with a none threshold to ensure that the continuous dice coefficient is calculated
    soft_metric = SoftDICE(name="DICE", along_axes=(0, 1), ignore_null=False)
    all_one_tenth_logits = 0.1 * torch.ones((10, 1, 10, 10, 10))
    soft_metric.update(all_one_tenth_logits, all_ones_targets)
    continuous_dice = soft_metric.compute()["DICE"]
    intersection = 100
    union = 0.5 * 1.1 * 1000
    dice_target = intersection / (union + 1e-7)
    pytest.approx(continuous_dice, abs=0.001) == dice_target


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
