import math

import numpy as np
import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import BinaryDice


def test_binary_dice_metric_1d_and_clear() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=None)

    # Test with two 1D examples
    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    dice.clear()

    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_1d_and_clear_with_no_label_dim() -> None:
    dice = BinaryDice(name="BinaryDice", batch_dim=None, zero_division=None)

    # Test with two 1D examples
    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    dice.clear()

    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_1d_accumulation() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=None)

    # Test accumulation
    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.5)

    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(12.0 / 18.0)


def test_binary_dice_metric_1d_accumulation_pos_label_0() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, pos_label=0, batch_dim=None, zero_division=None)

    # Test accumulation
    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 0, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(6 / (6 + 1 + 3))

    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(10 / (10 + 2 + 4))


def test_binary_dice_metric_3d() -> None:
    # Test higher dimension examples. Shape is (1, 2, 4)
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=None)
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)


def test_binary_dice_metric_zero_div_ignore() -> None:
    # Test that null scores are ignored when zero_division argument is set to None
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=None)

    # Initial test
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    # Test that ignore null is working by adding a null sample
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    dice.clear()

    # Test that NaN is returned when all dice coefficients are null. Should return NaN
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert isinstance(result["BinaryDice"], float)  # Appeases mypy, ensure its a float before checking if its NaN
    assert math.isnan(result["BinaryDice"])

    # Ensure that adding more to the score returns the correct value
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)


def test_binary_dice_metric_zero_div_1() -> None:
    # Test zero_division argument when a specific replacement score is specified.
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=1.0)

    # Initial test
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    # Test that ignore null is working by adding a null sample, which shouldn't move the needle, since there
    # will be no zero division (we have seen TPs and batch_dim is None)
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.8)

    dice.clear()

    # Since we've set the zero_division to 1.0, this means we'll get 1.0 for the score, because all we've seen are TNs
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(1.0)


def test_binary_dice_metric_accumulation() -> None:
    # Accumulate three times
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=None, zero_division=None)

    preds = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, t)
    result = dice.compute()
    assert result["BinaryDice"] == approx(10.0 / 12.0)

    preds = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, t)
    result = dice.compute()
    assert result["BinaryDice"] == approx(10.0 / 20.0)

    preds = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, t)
    result = dice.compute()
    assert result["BinaryDice"] == approx(20.0 / 32.0)


def test_binary_dice_with_high_dimensional_tensors() -> None:
    dice = BinaryDice(name="BinaryDice", batch_dim=None, label_dim=1, zero_division=1.0, threshold=0.5)

    predictions = torch.ones((10, 1, 10, 10, 10))
    targets = torch.ones((10, 1, 10, 10, 10))

    dice.update(predictions, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(1.0)

    dice.clear()

    # Test with random logits between 0 and 1
    np.random.seed(42)
    predictions = torch.Tensor(np.random.rand(10, 1, 10, 10, 10))

    dice.update(predictions, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.6598031841976006, abs=0.0001)

    dice.clear()

    predictions = torch.zeros((10, 1, 10, 10, 10))

    dice.update(predictions, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(0.0)

    dice.clear()

    # Test with union of zero to ensure edge case is 1.0 when zero division = 1.0
    predictions = torch.zeros((10, 1, 10, 10, 10))
    targets = torch.zeros((10, 1, 10, 10, 10))

    dice.update(predictions, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(1.0)

    # Test all TP's added
    targets = torch.ones((10, 1, 10, 10, 10))
    predictions = torch.ones((10, 1, 10, 10, 10))

    dice.update(predictions, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(1.0)

    # Change the threshold to 0.1
    dice = BinaryDice(name="BinaryDice", batch_dim=None, label_dim=1, zero_division=1.0, threshold=0.1)
    targets = torch.ones((10, 1, 10, 10, 10))
    predictions = torch.ones((10, 1, 10, 10, 10))
    assert dice(predictions, targets) == approx(1.0)

    # Test with 0.25 in all entries, but with a lower threshold for classification as 1
    predictions = 0.25 * torch.ones((10, 1, 10, 10, 10))
    assert dice(predictions, targets) == approx(1.0)

    # Test with a none threshold to ensure that the continuous dice coefficient is calculated
    dice = BinaryDice(name="BinaryDice", batch_dim=None)

    predictions = 0.1 * torch.ones((10, 1, 10, 10, 10))

    intersection = 100
    union = 0.5 * 1.1 * 1000
    dice_target = intersection / (union)
    assert dice(predictions, targets) == approx(dice_target)
