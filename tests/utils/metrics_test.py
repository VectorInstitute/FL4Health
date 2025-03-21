import math
import re

import numpy as np
import pytest
import torch
from pytest import approx
from torch.nn.functional import one_hot

from fl4health.utils.metrics import (
    F1,
    ROC_AUC,
    Accuracy,
    BalancedAccuracy,
    Dice,
    HardDice,
    MetricManager,
    Recall,
    align_pred_and_target_shapes,
)


def get_dummy_classification_tensors(
    ohe_shape: tuple[int, ...], class_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns 4 versions of the same dummy tensor to use for metric tests."""
    n_classes = ohe_shape[class_dim]
    assert n_classes > 1, "Must have at least 2 classes"

    # Create soft one-hot-encoded tensor
    soft_ohe = torch.rand(size=ohe_shape)
    soft_ohe = torch.softmax(soft_ohe, dim=class_dim)

    # Create hard not one-hot-encoded tensor
    hard_not_ohe = torch.argmax(soft_ohe, dim=class_dim)

    # Create hard one-hot-encoded tensor
    hard_ohe = torch.zeros(size=ohe_shape)
    hard_ohe_view = hard_not_ohe.view((*hard_not_ohe.shape[:class_dim], 1, *hard_not_ohe.shape[class_dim:]))
    hard_ohe.scatter_(class_dim, hard_ohe_view, 1)

    soft_not_ohe = torch.tensor([])
    if n_classes == 2:  # Only binary classification can have a continious tensor that is not one-hot-encoded.
        soft_not_ohe = torch.select(soft_ohe, class_dim, 1)

    return hard_ohe, soft_ohe, hard_not_ohe, soft_not_ohe


def test_multiclass_align() -> None:
    """Tests the auto shape alignment used by the ClassificationMetric base class for multiclass classification."""
    # Create dummy preds and targets
    hard_preds_ohe, soft_preds_ohe, hard_preds, _ = get_dummy_classification_tensors((2, 3, 5, 9, 3), 1)
    hard_targets_ohe, soft_targets_ohe, hard_targets, _ = get_dummy_classification_tensors((2, 3, 5, 9, 3), 1)

    # Test align on different combinations of mismatched shapes
    preds, targets = align_pred_and_target_shapes(soft_preds_ohe, hard_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, soft_preds_ohe).all()
    assert torch.isclose(targets, hard_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(hard_preds_ohe, hard_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, hard_preds_ohe).all()
    assert torch.isclose(targets, hard_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(hard_preds, hard_targets_ohe)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, hard_preds_ohe).all()
    assert torch.isclose(targets, hard_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(hard_preds, soft_targets_ohe)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, hard_preds_ohe).all()
    assert torch.isclose(targets, soft_targets_ohe).all()

    # Test that if shapes are the same outputs are unchanged
    preds, targets = align_pred_and_target_shapes(hard_preds_ohe, soft_targets_ohe)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, hard_preds_ohe).all()
    assert torch.isclose(targets, soft_targets_ohe).all()

    # Test that if shapes are the same outputs are unchanged
    preds, targets = align_pred_and_target_shapes(soft_preds_ohe, hard_targets_ohe)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, soft_preds_ohe).all()
    assert torch.isclose(targets, hard_targets_ohe).all()


def test_binary_align() -> None:
    """Tests the auto shape alignment used by the ClassificationMetric base class for binary classification."""
    # Create dummy preds and targets. H stands for 'hard' and s stands for 'soft'
    h_preds_ohe, s_preds_ohe, h_preds, s_preds = get_dummy_classification_tensors((4, 2, 5, 9, 3), 1)
    h_targets_ohe, s_targets_ohe, h_targets, s_targets = get_dummy_classification_tensors((4, 2, 5, 9, 3), 1)

    # Test align with soft tensors that are not one-hot-encoded
    preds, targets = align_pred_and_target_shapes(s_preds, h_targets_ohe)
    assert preds.shape == targets.shape == (4, 2, 5, 9, 3)
    assert torch.isclose(preds, s_preds_ohe).all()
    assert torch.isclose(targets, h_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(s_preds, s_targets_ohe)
    assert preds.shape == targets.shape == (4, 2, 5, 9, 3)
    assert torch.isclose(preds, s_preds_ohe).all()
    assert torch.isclose(targets, s_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(h_preds_ohe, s_targets)
    assert preds.shape == targets.shape == (4, 2, 5, 9, 3)
    assert torch.isclose(preds, h_preds_ohe).all()
    assert torch.isclose(targets, s_targets_ohe).all()

    preds, targets = align_pred_and_target_shapes(s_preds_ohe, s_targets)
    assert preds.shape == targets.shape == (4, 2, 5, 9, 3)
    assert torch.isclose(preds, s_preds_ohe).all()
    assert torch.isclose(targets, s_targets_ohe).all()

    # If shapes of the input tensors are the same then the outputs should be as well
    preds, targets = align_pred_and_target_shapes(s_preds, s_targets)
    assert preds.shape == targets.shape == (4, 5, 9, 3)
    assert torch.isclose(preds, s_preds).all()
    assert torch.isclose(targets, s_targets).all()


def test_align_exceptions() -> None:
    """Tests that the proper exceptions are raised when attempting to align pred and target shapes."""
    # Define a pattern to ensure that the exception has something to do with inferring the channel dim.
    multiple_axes_pattern = re.compile("Found multiple axes that could be the label dimension", flags=re.IGNORECASE)
    same_shape_different_size_patter = re.compile(
        "The inferred candidate dimension has different sizes on each tensor, " "was expecting one to be empty.",
        flags=re.IGNORECASE,
    )
    axes_of_same_size_pattern = re.compile(
        "A dimension adjacent to the label dimension appears to have the same size.", flags=re.IGNORECASE
    )

    # Channel dim can not be resolved if shapes differ in more than 1 dimension
    hard_preds_ohe, soft_preds_ohe, _, _ = get_dummy_classification_tensors((2, 3, 5, 9, 3), 1)
    hard_targets_ohe, _, hard_targets, _ = get_dummy_classification_tensors((2, 3, 5, 9, 6), 1)

    with pytest.raises(Exception, match=multiple_axes_pattern):
        align_pred_and_target_shapes(soft_preds_ohe, hard_targets)

    with pytest.raises(Exception, match=same_shape_different_size_patter):
        align_pred_and_target_shapes(hard_preds_ohe, hard_targets_ohe)

    # Channel dim can not be resolved if the dimension directly afterwards has the same size
    hard_preds_ohe, soft_preds_ohe, _, _ = get_dummy_classification_tensors((2, 3, 3, 9, 6), 1)
    hard_targets_ohe, _, hard_targets, _ = get_dummy_classification_tensors((2, 3, 3, 9, 6), 1)

    with pytest.raises(Exception, match=axes_of_same_size_pattern):
        align_pred_and_target_shapes(soft_preds_ohe, hard_targets)


def test_hard_dice_metric_1d_and_clear() -> None:
    """Simple test of HardDice metric for 1D predictions."""
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=None, zero_division=None)

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
    """Test that 1D predictions are properly accumulated."""
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=None, zero_division=None)

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
    """Test that HardDice works with data that has more than 1 dimension."""
    # Test higher dimension examples. Shape is (1, 2, 4)
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=None, zero_division=None)
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(0.8)


def test_hard_dice_metric_zero_div_ignore() -> None:
    """Test that null scores are ignored when zero_division argument is set to None."""
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=None, zero_division=None)

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


def test_hard_dice_metric_zero_div_1() -> None:
    """Test zero_division argument when a specific replacement score is specified."""
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=None, zero_division=1.0)

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
    """Test ignore_background argument of HardDice metric"""
    # Test ignore background
    hd = HardDice(name="DICE", along_axes=(0,), ignore_background=1, zero_division=None)
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
    """Test the along_axes argument of the HardDice metric."""
    # Test computing dice coefficients along different axes. Shape (n, 2, 4)
    hd = HardDice(name="DICE", along_axes=(1,), ignore_background=None, zero_division=None)
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
    hd = HardDice(name="DICE", along_axes=(2,), ignore_background=None, zero_division=None)
    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["DICE"] == approx(5 / 6)  # (1 + 2/3 + 2/3 + 1) / 4

    # Test using along multiple axes. Shape (3, 2, 4)
    hd = HardDice(name="DICE", along_axes=(0, 1), ignore_background=None, zero_division=None)
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
    accuracy_metric = Accuracy(along_axes=(0,), exact_match=True)

    pred1 = one_hot(torch.arange(0, 6) % 5)
    target1 = torch.tensor([0, 1, 2, 3, 4, 0])
    accuracy_metric.update(pred1, target1)
    result1 = accuracy_metric.compute()
    assert result1["Accuracy"] == approx(1.0)

    accuracy_metric.clear()
    pred2 = one_hot(torch.arange(0, 4) % 3)
    target2 = torch.tensor([0, 1, 2, 2])
    accuracy2 = accuracy_metric(pred2, target2)
    assert accuracy2 == 0.75


def test_accuracy_accumulation_and_correctness() -> None:
    # This implements the traditional accuracy metric, even for multi-class problems
    accuracy_metric_1 = Accuracy(along_axes=(0,), exact_match=True)
    # This implements macro-averaging of class-by-class accuracy, which is often much higher than raw accuracy
    # because it counts true negative predictions, which are more prevalent as the number of classes grows
    accuracy_metric_2 = Accuracy(along_axes=(1,), exact_match=False)

    pred1 = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    target1 = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    accuracy_metric_1.update(pred1, target1)
    accuracy_metric_2.update(pred1, target1)

    pred2 = torch.tensor([[1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
    target2 = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    accuracy_metric_1.update(pred2, target2)
    accuracy_metric_2.update(pred2, target2)

    result1 = accuracy_metric_1.compute()
    result2 = accuracy_metric_2.compute()

    assert result1["Accuracy"] == approx(0.5)
    assert result2["Accuracy"] == approx(0.6667, abs=1e-4)

    accuracy_metric_1.clear()
    accuracy_metric_2.clear()

    # In the binary label case, these should be equal because incorrect or correct for one class implies the same for
    # the other class.

    pred1 = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    target1 = torch.tensor(
        [
            [0, 1],
            [
                1,
                0,
            ],
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    accuracy_metric_1.update(pred1, target1)
    accuracy_metric_2.update(pred1, target1)

    pred2 = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]])
    target2 = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 0], [0, 1]])
    accuracy_metric_1.update(pred2, target2)
    accuracy_metric_2.update(pred2, target2)

    result1 = accuracy_metric_1.compute()
    result2 = accuracy_metric_2.compute()

    assert result1["Accuracy"] == approx(0.6)
    assert result2["Accuracy"] == approx(0.6)

    accuracy_metric_1.clear()
    accuracy_metric_2.clear()

    # Each should handle the multi-label, multi-class setting as well. Exact match would only consider those tuples
    # that were FULL matches, i.e. all predictions matched all labels for each example.
    # Binarize must be set to None so that it predictions as they are classes
    accuracy_metric_1 = Accuracy(along_axes=(0,), exact_match=False, binarize=None)
    accuracy_metric_2 = Accuracy(along_axes=(1,), exact_match=False, binarize=None)

    pred1 = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    target1 = torch.tensor([[0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0]])
    accuracy_metric_1.update(pred1, target1)
    accuracy_metric_2.update(pred1, target1)

    pred2 = torch.tensor([[1, 1, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
    target2 = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    accuracy_metric_1.update(pred2, target2)
    accuracy_metric_2.update(pred2, target2)

    result1 = accuracy_metric_1.compute()
    result2 = accuracy_metric_2.compute()

    assert result1["Accuracy"] == approx(0.6667, abs=1e-4)
    assert result2["Accuracy"] == approx(0.6667, abs=1e-4)


def test_binary_accuracy() -> None:
    accuracy_metric = Accuracy(binarize=None)

    pred1 = torch.Tensor([0, 1, 1, 0, 1])
    target1 = torch.Tensor([0, 0, 1, 1, 1])
    accuracy1 = accuracy_metric(pred1, target1)
    assert accuracy1 == approx(3.0 / 5.0)


def test_binary_recall() -> None:
    # Computes the standard recall for binary classification
    r1 = Recall("recall", along_axes=(0,), binarize=None)
    pred1 = torch.Tensor([0, 1, 1, 0, 1])
    target1 = torch.Tensor([0, 0, 1, 1, 1])
    r1.update(pred1, target1)
    result = r1.compute()
    assert result["recall"] == approx(2.0 / 3.0, abs=1e-4)

    # This treats positive predictions that line up with a label as a TP, regardless of class. So there are 3 TPs
    # for the provided input and 2 FNs for recall computed as TP/(TP+FN)
    r2 = Recall("recall", along_axes=(0,), binarize=None)
    pred1 = torch.Tensor([[1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
    target1 = torch.Tensor([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    r2.update(pred1, target1)
    result = r2.compute()
    assert result["recall"] == approx(3.0 / 5.0, abs=1e-4)


def test_balanced_accuracy() -> None:
    metric = BalancedAccuracy()

    logits = torch.Tensor([[0.75, 0.25], [0.12, 0.88], [0.9, 0.1], [0.94, 0.06], [0.78, 0.22], [0.08, 0.92]])
    target = torch.Tensor([0, 1, 0, 0, 1, 0])

    metric.update(logits, target)
    result = metric.compute()
    assert result["balanced_accuracy"] == approx(0.625)

    metric.clear()
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
    metric.update(logits, target)
    result = metric.compute()
    assert result["balanced_accuracy"] == approx(0.5)


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
    metric = F1(along_axes=(1,), binarize=1, weighted=True)

    logits = torch.Tensor(
        [
            [3.0, 1.0, 2.0],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 1.1],
            [0.5, 3.0, 1.5],
        ]
    )
    target = torch.Tensor([0, 0, 2, 0, 2])
    metric.update(logits, target)
    result = metric.compute()
    assert result["F1"] == approx(0.68)


def test_hard_dice_default_threshold() -> None:
    """Essentially a copy of a test for the old Dice metric class to ensure the new one can act as a stand in."""
    # Test with the default spatial and epsilon values
    metric = HardDice(name="DICE", along_axes=(0, 1), zero_division=1.0, binarize=0.5)
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

    # Test with intersection of zero to ensure edge case is isclose to 0.0
    metric.clear()
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    metric.update(all_zeros_logits, all_ones_targets)
    dice_intersection_zero = metric.compute()["DICE"]
    pytest.approx(dice_intersection_zero, abs=0.000001) == 0.0

    # Test with union of zero to ensure edge case is isclose to 1.0
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
    """Essentially a copy of a test for the old Dice metric class to ensure the new one can act as a stand in."""
    # Change the threshold to 0.1
    metric = HardDice(name="DICE", along_axes=(0, 1), zero_division=1.0, binarize=0.1)

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
    soft_metric = Dice(name="DICE", along_axes=(0, 1), zero_division=1.0)
    all_one_tenth_logits = 0.1 * torch.ones((10, 1, 10, 10, 10))
    soft_metric.update(all_one_tenth_logits, all_ones_targets)
    continuous_dice = soft_metric.compute()["DICE"]
    intersection = 100
    union = 0.5 * 1.1 * 1000
    dice_target = intersection / (union + 1e-7)
    pytest.approx(continuous_dice, abs=0.001) == dice_target


def test_metric_accumulation() -> None:
    a = Accuracy(along_axes=(0,), exact_match=True)

    pred1 = one_hot(torch.arange(4) % 3)
    pred2 = one_hot(torch.arange(4) % 3)
    pred3 = one_hot(torch.arange(4) % 3)

    target1 = torch.tensor([0, 1, 2, 0])  # 1.0
    target2 = torch.tensor([2, 0, 1, 2])  # 0.0
    target3 = torch.tensor([0, 1, 1, 1])  # 0.5

    preds = [pred1, pred2, pred3]
    targets = [target1, target2, target3]

    for pred, target in zip(preds, targets):
        a.update(pred, target)

    assert a.compute()["Accuracy"] == 0.5

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

    mm = MetricManager(
        [F1(along_axes=(1,), binarize=1, weighted=True), Accuracy(along_axes=(0,), binarize=1, exact_match=True)],
        "test",
    )

    for logits, target in zip(logits_list, target_list):
        preds = {"prediction": logits}
        mm.update(preds, target)
    metrics = mm.compute()

    assert metrics["test - prediction - F1"] == pytest.approx(0.80285714285, abs=0.00001)
    assert metrics["test - prediction - Accuracy"] == approx(0.8)
