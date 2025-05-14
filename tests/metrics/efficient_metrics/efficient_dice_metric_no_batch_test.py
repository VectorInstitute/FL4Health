import math

import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import MultiClassDice
from fl4health.utils.random import set_all_random_seeds


def test_multi_dice_metric_with_threshold() -> None:
    set_all_random_seeds(42)

    dice = MultiClassDice(name="DICE", label_dim=2, batch_dim=None, threshold=2, zero_division=None)

    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # TP = [1, 2, 0], TN = [3, 3, 2], FP = [1, 1, 1], FN = [1, 0, 3]
    # Dice for each class is 2.0/4.0, 4.0/5.0, 0.0/4.0. So the final score should be the average
    assert result["DICE"] == approx((1.0 / 3.0) * (2.0 / 4.0 + 4.0 / 5.0 + 0.0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # For this batch, TP = [1, 3, 1], TN = [3, 1, 3], FP = [0, 1, 0], FN = [2, 1, 2]
    # Adding these to the above results in Dice per class of [4/(4+1+3), 10/(10 + 2 + 1), 2/(2 + 1 + 5)]
    assert result["DICE"] == approx((1.0 / 3.0) * (4.0 / 8.0 + 10 / 13.0 + 2.0 / 8.0))

    # Test Dropping zero divisions
    # Kill the threshold so we can have all zeros for preds
    dice.threshold = None
    logits = torch.zeros((2, 3, 3))
    targets = torch.zeros((2, 3, 3))

    dice.update(logits, targets)
    result = dice.compute()
    # Should be the same since all added pieces have zero division and are ignored
    assert result["DICE"] == approx((1.0 / 3.0) * (4.0 / 8.0 + 10 / 13.0 + 2.0 / 8.0))

    # Move to zero division = 1.0 such that each of these examples adds 1.0 to the Dice. Since there are no zero
    # divisions, this should still have no effect on the Dice
    dice.zero_division = 1.0
    result = dice.compute()
    assert result["DICE"] == approx((1.0 / 3.0) * (4.0 / 8.0 + 10 / 13.0 + 2.0 / 8.0))

    dice.zero_division = None
    dice.clear()

    # Test NaN
    logits = torch.zeros((2, 3, 3))
    targets = torch.zeros((2, 3, 3))

    dice.update(logits, targets)
    result = dice.compute()
    assert isinstance(result["DICE"], float)  # Appeases mypy, ensure its a float before checking if its NaN
    assert math.isnan(result["DICE"])


def test_multi_dice_metric_ignore_background() -> None:
    set_all_random_seeds(42)

    dice = MultiClassDice(
        name="DICE", label_dim=2, batch_dim=None, threshold=2, ignore_background=2, zero_division=None
    )

    # Test ignore background
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # TP = [1, 2, 0], TN = [3, 3, 2], FP = [1, 1, 1], FN = [1, 0, 3].
    # However, the first class is being ignored.
    # Dice for other classes is 4.0/5.0, 0.0/4.0. So the final score should be the average
    assert result["DICE"] == approx(0.5 * (4.0 / 5.0 + 0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    # For this batch, TP = [1, 3, 1], TN = [3, 1, 3], FP = [0, 1, 0], FN = [2, 1, 2]
    # However, the first class is being ignored.
    # Adding these to the above results in Dice per class of [10/(10 + 2 + 1), 2/(2 + 1 + 5)]
    result = dice.compute()
    assert result["DICE"] == approx(0.5 * (10.0 / 13.0 + 2.0 / 8.0))


def test_continuous_multi_dice_metric() -> None:
    set_all_random_seeds(42)

    # Test ignore background
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice = MultiClassDice(name="DICE", label_dim=2, batch_dim=None, zero_division=None)
    dice.update(logits, targets)
    result = dice.compute()
    tp1 = [0.8694 + 0.4294, 0.9150 + 0.9346, 0.6009 + 0.7411 + 0.5739]
    fp1 = [0.8823 + 0.9593 + 0.2566 + 0.1332, 0.3904 + 0.7936 + 0.5677 + 0.8854, 0.3829 + 0.9408 + 0.5936]
    fn1 = [(1 - 0.8694) + (1 - 0.4294), (1 - 0.9150) + (1 - 0.9346), (1 - 0.6009) + (1 - 0.7411) + (1 - 0.5739)]
    dice_target_1 = (2 * tp1[0]) / (2 * tp1[0] + fp1[0] + fn1[0])
    dice_target_2 = (2 * tp1[1]) / (2 * tp1[1] + fp1[1] + fn1[1])
    dice_target_3 = (2 * tp1[2]) / (2 * tp1[2] + fp1[2] + fn1[2])
    assert result["DICE"] == approx((1.0 / 3.0) * (dice_target_1 + dice_target_2 + dice_target_3), abs=1e-4)

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    tp2 = [0.5779 + 0.7104 + 0.7539, 0.9040 + 0.6343 + 0.7886 + 0.1165, 0.5547 + 0.7890 + 0.9103]
    fp2 = [0.3423 + 0.2814 + 0.3068, 0.9464 + 0.1952, 0.3644 + 0.5895 + 0.0050]
    fn2 = [
        (1 - 0.5779) + (1 - 0.7104) + (1 - 0.7539),
        (1 - 0.9040) + (1 - 0.6343) + (1 - 0.7886) + (1 - 0.1165),
        (1 - 0.5547) + (1 - 0.7890) + (1 - 0.9103),
    ]
    dice_target_1 = (2 * (tp1[0] + tp2[0])) / (2 * (tp1[0] + tp2[0]) + fp1[0] + fn1[0] + fp2[0] + fn2[0])
    dice_target_2 = (2 * (tp1[1] + tp2[1])) / (2 * (tp1[1] + tp2[1]) + fp1[1] + fn1[1] + fp2[1] + fn2[1])
    dice_target_3 = (2 * (tp1[2] + tp2[2])) / (2 * (tp1[2] + tp2[2]) + fp1[2] + fn1[2] + fp2[2] + fn2[2])
    assert result["DICE"] == approx((1.0 / 3.0) * (dice_target_1 + dice_target_2 + dice_target_3), abs=1e-4)


def test_more_spatial_dimensions() -> None:
    dice = MultiClassDice(name="DICE", label_dim=1, batch_dim=None)

    predictions = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for predictions
    predictions[:, 1, :, :] = 0

    dice_coefficient = dice(predictions, targets)
    # Dice for the first label should be 1.0 and the second should be 0. Thus the final score should be 0.5
    assert dice_coefficient == approx(0.5)

    # Set the second label to be all TNs to make sure we're properly ignoring those contributions to the dice
    predictions = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    predictions[:, 1, :, :] = 0
    targets[:, 1, :, :] = 0

    dice_coefficient = dice(predictions, targets)
    # Dice for the first label should be 1.0 and the second should be ignored, since zero_division is None
    assert dice_coefficient == approx(1.0)

    # change threshold to be 1 and zero division to be 1.0
    dice = MultiClassDice(name="DICE", label_dim=0, zero_division=1.0, threshold=0.1, batch_dim=None)

    predictions = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    predictions[:, 1, :, :] = 0
    targets[:, 1, :, :] = 0

    dice_coefficient = dice(predictions, targets)
    # Dice for the first label should be 1.0 and the second should be 1.0, since we have all TNs for that channel
    assert dice_coefficient == approx(1.0)
