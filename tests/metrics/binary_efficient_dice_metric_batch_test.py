import math

import numpy as np
import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import BinaryDice


def test_binary_dice_metric_1d_and_clear() -> None:
    """Simple test of Dice metric for 1D predictions."""
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Test with two 1D examples
    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(2.0 / 3.0)

    hd.clear()  # Clear to restart
    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)


def test_binary_dice_metric_1d_and_clear_with_no_label_dim() -> None:
    """Simple test of Dice metric for 1D predictions."""
    hd = BinaryDice(name="BinaryDice", batch_dim=1, zero_division=None)

    # Test with two 1D examples
    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(2.0 / 3.0)

    hd.clear()  # Clear to restart
    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)


def test_binary_dice_metric_1d_with_no_label_dim_batch_dim_0() -> None:
    hd = BinaryDice(name="BinaryDice", batch_dim=0, zero_division=None)

    # Test with two 1D examples
    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    assert result["BinaryDice"] == approx(0.8)

    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Second example has a Dice of 0.5 and we should be averaging the results with the current batch dim
    assert result["BinaryDice"] == approx((0.5 + 0.8) / 2.0)


def test_binary_dice_metric_1d_accumulation() -> None:
    """Test that 1D predictions are properly accumulated."""
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Test accumulation
    p = torch.tensor([[0, 0, 0, 1, 0, 0, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    assert result["BinaryDice"] == approx(1.0 / 3.0)

    p = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    t = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 1]])
    hd.update(p, t)
    result = hd.compute()
    # Because zero division is None we ignore those examples with only TNs (first two examples)
    # With batch_dim = 1, we accumulate 6 examples of Dice 1.0 and 6 with Dice of 0.0. So should be 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_2d() -> None:
    """Test that Dice works with data that has more than 1 dimension."""
    # Test higher dimension examples. Shape is (1, 2, 4)
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_2d_with_zero_div() -> None:
    """Test that Dice works with data that has more than 1 dimension."""
    # Test higher dimension examples. Shape is (1, 2, 4)
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)
    p = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 0, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example is all TNs, with zero_division, this is ignored. Second example is all 1s, so Dice 1.0
    assert result["BinaryDice"] == approx(1.0)


def test_binary_dice_metric_zero_div_ignore() -> None:
    """Test that null scores are ignored when zero_division argument is set to None."""
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)

    # Initial test
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)

    # Test that ignore null is working by adding a null sample
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["BinaryDice"] == approx(0.5)

    # Test that NaN is returned when all dice coefficients are null. Should return NaN
    hd.clear()
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert isinstance(result["BinaryDice"], float)  # Appeases mypy, ensure its a float before checking if its NaN
    assert math.isnan(result["BinaryDice"])

    # Ensure that adding more to the score returns the correct value
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)


def test_binary_dice_metric_zero_div_1() -> None:
    """Test zero_division argument when a specific replacement score is specified."""
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=1.0)

    # Initial test
    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 0 TPs -> 0 Dice, second is all TPs -> 1 Dice, so the average is 0.5
    assert result["BinaryDice"] == approx(0.5)

    # Test that ignore null is working by adding a null sample with batch defined should produce 1.0 Dice for these
    # examples, which implies that we have 1 example with Dice 0, and now three examples of 1.0
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["BinaryDice"] == approx(3.0 / 4.0)

    # Since we've set the zero_division to 1.0, this means we'll get 1.0 for the score.
    hd.clear()
    p = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0]]])
    hd.update(p, t)
    result = hd.compute()
    assert result["BinaryDice"] == approx(1)


def test_binary_dice_metric_accumulation() -> None:
    # Accumulate three times
    hd = BinaryDice(name="BinaryDice", label_dim=0, batch_dim=1, zero_division=None)
    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 1 TP, 1 FN, 1 FP -> Dice 0.5, second has Dice 1.0
    assert result["BinaryDice"] == approx(1.5 / 2.0)

    p = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # Both of these examples have Dice = 0.0
    assert result["BinaryDice"] == approx(1.5 / 4.0)

    p = torch.tensor([[[0, 1, 0, 1], [1, 1, 1, 1]]])
    t = torch.tensor([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    hd.update(p, t)
    result = hd.compute()
    # First example has 1 TP, 1 FN, 1 FP -> Dice 0.5, second has Dice 1.0
    assert result["BinaryDice"] == approx(3.0 / 6.0)


def test_binary_dice_default_threshold() -> None:
    """Essentially a copy of a test for the old Dice metric class to ensure the new one can act as a stand in."""
    metric = BinaryDice(name="BinaryDice", batch_dim=0, label_dim=1, zero_division=1.0, threshold=0.5)
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    metric.update(all_ones_logits, all_ones_targets)
    dice_of_one = metric.compute()["BinaryDice"]
    approx(dice_of_one) == 1.0

    # Test with random logits between 0 and 1
    metric.clear()
    np.random.seed(42)
    random_logits = torch.Tensor(np.random.rand(10, 1, 10, 10, 10))
    metric.update(random_logits, all_ones_targets)
    random_dice = metric.compute()["BinaryDice"]
    # Because the images are all the same size (10, 10, 10) and targets are all 1s, there are no TNs and this makes
    # the average equivalent to the batch_dim = None case
    approx(random_dice, abs=0.00001) == 0.6598031841976006

    metric.clear()
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    metric.update(all_zeros_logits, all_ones_targets)
    dice_intersection_zero = metric.compute()["BinaryDice"]
    approx(dice_intersection_zero) == 0.0

    # Test with union of zero to ensure edge case is 1.0 when zero division = 1.0
    metric.clear()
    all_zeros_logits = torch.zeros((10, 1, 10, 10, 10))
    all_zeros_target = torch.zeros((10, 1, 10, 10, 10))
    metric.update(all_zeros_logits, all_zeros_target)
    dice_union_zero = metric.compute()["BinaryDice"]
    approx(dice_union_zero) == 1.0

    # Test all TP's added
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    metric.update(all_ones_logits, all_ones_targets)
    dice_of_one = metric.compute()["BinaryDice"]
    approx(dice_of_one) == 1.0

    # Change the threshold to 0.1
    metric = BinaryDice(name="BinaryDice", batch_dim=0, label_dim=1, zero_division=1.0, threshold=0.1)
    all_ones_targets = torch.ones((10, 1, 10, 10, 10))
    all_ones_logits = torch.ones((10, 1, 10, 10, 10))
    dice_of_one = metric(all_ones_logits, all_ones_targets)
    approx(dice_of_one) == 1.0

    # Test with 0.25 in all entries, but with a lower threshold for classification as 1
    all_one_quarter_logits = 0.25 * torch.ones((10, 1, 10, 10, 10))
    dice_of_one = metric(all_one_quarter_logits, all_ones_targets)
    approx(dice_of_one) == 1.0

    # Test with a none threshold to ensure that the continuous dice coefficient is calculated
    metric = BinaryDice(name="BinaryDice", batch_dim=0)
    all_one_tenth_logits = 0.1 * torch.ones((10, 1, 10, 10, 10))
    continuous_dice = metric(all_one_tenth_logits, all_ones_targets)
    # TPs is 0.1 * 1000, FP = 0.0, FNs = 0.9 * 10000 for all 10 images
    dice_target = 2 * 0.1 * 1000 / (2 * 0.1 * 1000 + 0.9 * 1000)
    approx(continuous_dice) == dice_target
