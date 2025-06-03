import math

import numpy as np
import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import BinaryDice


def test_binary_dice_metric_1d_and_clear() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Test with two 1D examples
    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(2.0 / 3.0)

    dice.clear()

    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)


def test_binary_dice_metric_1d_and_clear_with_no_label_dim() -> None:
    dice = BinaryDice(name="BinaryDice", batch_dim=1, zero_division=None)

    # Test with two 1D examples
    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(2.0 / 3.0)

    dice.clear()
    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)


def test_binary_dice_metric_1d_with_no_label_dim_batch_dim_0() -> None:
    dice = BinaryDice(name="BinaryDice", batch_dim=0, zero_division=None)

    # Test with two 1D examples
    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because batch_dim = 0, we accumulate all predictions across the batch size of 1
    assert result["BinaryDice"] == approx(0.8)

    # Accumulate more predictions and targets
    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Second example has a Dice of 0.5 and we should be averaging the results with the current batch dim
    assert result["BinaryDice"] == approx((0.5 + 0.8) / 2.0)


def test_binary_dice_metric_1d_accumulation_with_label_and_batch() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Test accumulation
    preds = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)

    preds = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    targets = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])

    dice.update(preds, targets)
    result = dice.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    # With batch_dim = 1, we accumulate 6 examples of Dice 1.0 and 6 with Dice of 0.0. So should be 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_3d() -> None:
    # Test higher dimension examples. Shape is (1, 2, 4)
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_3d_with_zero_div() -> None:
    # Test higher dimension examples. Shape is (1, 2, 4)
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    preds = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example is all TNs, with zero_division, this is ignored. Second example is all 1s, so Dice 1.0
    assert result["BinaryDice"] == approx(1.0)


def test_binary_dice_metric_3d_with_zero_div_accumulation() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Initial test
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)

    # Test that ignore null is working by adding a null sample
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    # Since we're adding a null example for all instances, we should ignore them and get the same Dice
    assert result["BinaryDice"] == approx(0.5)

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
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_zero_div_1() -> None:
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=1.0)

    # Initial test
    preds = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)

    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    # Test that ignore null is working by adding a null sample with batch defined should produce 1.0 Dice for these
    # examples, which implies that we have 1 example with Dice 0, and now three examples of 1.0
    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(3.0 / 4.0)

    dice.clear()

    # Since we've set the zero_division to 1.0, this means we'll get 1.0 for the score.
    preds = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])

    dice.update(preds, targets)
    result = dice.compute()
    assert result["BinaryDice"] == approx(1.0)


def test_binary_dice_metric_accumulation() -> None:
    # Accumulate three times
    dice = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)
    preds = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example has 1 TP, 1 FN, 1 FP -> Dice 0.5, second has Dice 1.0
    assert result["BinaryDice"] == approx(1.5 / 2.0)

    preds = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]]])
    targets = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # Both of these examples have Dice = 0.0
    assert result["BinaryDice"] == approx(1.5 / 4.0)

    preds = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    targets = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])

    dice.update(preds, targets)
    result = dice.compute()
    # First example has 1 TP, 1 FN, 1 FP -> Dice 0.5, second has Dice 1.0
    assert result["BinaryDice"] == approx(3.0 / 6.0)


def test_binary_dice_in_high_dimensions() -> None:
    dice = BinaryDice(name="BinaryDice", batch_dim=0, label_dim=1, zero_division=1.0, threshold=0.5)

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
    # Because the images are all the same size (10, 10, 10) and targets are all 1s, there are no TNs and this makes
    # the average equivalent to the batch_dim = None case
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
    dice = BinaryDice(name="BinaryDice", batch_dim=0, label_dim=1, zero_division=1.0, threshold=0.1)

    targets = torch.ones((10, 1, 10, 10, 10))
    predictions = torch.ones((10, 1, 10, 10, 10))

    assert dice(predictions, targets) == approx(1.0)

    # Test with 0.25 in all entries, but with a lower threshold for classification as 1
    predictions = 0.25 * torch.ones((10, 1, 10, 10, 10))
    assert dice(predictions, targets) == approx(1.0)

    # Test with a none threshold to ensure that the continuous dice coefficient is calculated
    dice = BinaryDice(name="BinaryDice", batch_dim=0)

    predictions = 0.1 * torch.ones((10, 1, 10, 10, 10))

    # TPs is 0.1 * 1000, FP = 0.0, FNs = 0.9 * 10000 for all 10 images
    dice_target = 2 * 0.1 * 1000 / (2 * 0.1 * 1000 + 0.9 * 1000)
    assert dice(predictions, targets) == approx(dice_target)
