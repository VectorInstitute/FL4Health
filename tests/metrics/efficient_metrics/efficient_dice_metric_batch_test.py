import math

import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import MultiClassDice
from fl4health.utils.random import set_all_random_seeds


def test_multi_dice_metric_with_threshold() -> None:
    set_all_random_seeds(42)

    dice = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, threshold=2, zero_division=None)

    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # Dice scores collapse to D = [[0, 2/(2), 0], [2/(2+1), 2/(2+1), 0]] per instance (2), class (3)
    assert result["DICE"] == approx((1.0 / 6.0) * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # Dice scores collapse to D = [[0, 4/(4+1), 0], [2/(2), 2/(2+1), 2/2]] per instance (2), class (3) in this batch
    # Adding these to the above and averaging yields
    assert result["DICE"] == approx(
        (1.0 / 12.0)
        * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 2.0 + 2.0 / 3.0 + 2.0 / 2.0)
    )

    # Test Dropping zero divisions
    # Kill the threshold so we can have all zeros for preds
    dice.threshold = None
    logits = torch.zeros((2, 3, 3))
    targets = torch.zeros((2, 3, 3))

    dice.update(logits, targets)
    result = dice.compute()
    # Should be the same since all added pieces are just TNs and are ignored
    assert result["DICE"] == approx(
        (1.0 / 12.0)
        * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 2.0 + 2.0 / 3.0 + 2.0 / 2.0)
    )

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

    dice = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, threshold=2, ignore_background=2, zero_division=None)

    # Test ignore background
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # Dice scores collapse to D = [[0, 2/(2), 0], [2/(2+1), 2/(2+1), 0]] per instance (2), class (3)
    # However, we're dropping the first class using ignore background
    assert result["DICE"] == approx((1.0 / 4.0) * (1.0 + 0 + 2.0 / 3.0 + 0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    # Dice scores collapse to D = [[0, 4/(4+1), 0], [2/(2), 2/(2+1), 2/2]] per instance (2), class (3) in this batch
    # However, we're dropping the first class using ignore background
    # Adding these to the above and averaging yields
    assert result["DICE"] == approx((1.0 / 8.0) * (1.0 + 0 + 2.0 / 3.0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 3.0 + 2.0 / 2.0))


def test_continuous_multi_dice_metric() -> None:
    set_all_random_seeds(42)

    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, zero_division=None)
    dice.update(logits, targets)
    result = dice.compute()
    tp1 = [[0, 0.9150, 0.6009], [0.8694 + 0.4294, 0.9346, 0.7411 + 0.5739]]
    fp1 = [[0.8823 + 0.9593 + 0.2566, 0.3904 + 0.7936, 0.3829 + 0.9408], [0.1332, 0.5677 + 0.8854, 0.5936]]
    fn1 = [[0, (1 - 0.9150), (1 - 0.6009)], [(1 - 0.8694) + (1 - 0.4294), (1 - 0.9346), (1 - 0.7411) + (1 - 0.5739)]]

    dice_target_1 = [
        [(2 * tp1[i][j]) / (2 * tp1[i][j] + fp1[i][j] + fn1[i][j]) for j in range(len(tp1[i]))]
        for i in range(len(tp1))
    ]
    assert result["DICE"] == approx((1.0 / 6.0) * sum(sum(d) for d in dice_target_1), abs=1e-4)

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    dice.update(logits, targets)
    result = dice.compute()
    tp2 = [[0.5779 + 0.7104, 0.9040 + 0.6343, 0.5547 + 0.7890], [0.9103, 0.7886 + 0.1165, 0.7539]]
    fp2 = [[0.3423, 0.9464, 0.3644], [0.2814 + 0.3068, 0.1952, 0.5895 + 0.0050]]
    fn2 = [
        [(1 - 0.5779) + (1 - 0.7104), (1 - 0.9040) + (1 - 0.6343), (1 - 0.5547) + (1 - 0.7890)],
        [(1 - 0.9103), (1 - 0.7886) + (1 - 0.1165), (1 - 0.7539)],
    ]

    dice_target_2 = [
        [(2 * tp2[i][j]) / (2 * tp2[i][j] + fp2[i][j] + fn2[i][j]) for j in range(len(tp2[i]))]
        for i in range(len(tp2))
    ]
    assert result["DICE"] == approx((1.0 / 12.0) * sum(sum(d) for d in (dice_target_1 + dice_target_2)), abs=1e-4)


def test_more_spatial_dimensions() -> None:
    dice = MultiClassDice(name="DICE", label_dim=1, batch_dim=2)

    preds = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero
    preds[:, 1, :, :] = 0

    dice_coefficient = dice(preds, targets)
    # Dice for the first label should be 1.0 for all batch components and the second should be 0. Thus the final score
    # should be 0.5
    assert dice_coefficient == approx(0.5)

    preds = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel of predictions to zero
    preds[:, 1, :, :] = 0
    # Set entries in the second channel of targets to zero for the first 5 instances
    targets[:, 1, 0:5, :] = 0

    dice_coefficient = dice(preds, targets)
    # Dice for the first label should be 1.0 for all batch components and the second should be 0 for the final five
    # entries and the first 5 entries should be ignored since zero_division = None and they are all TNs
    assert dice_coefficient == approx((1.0 / 15.0) * (10 * 1.0 + 5 * 0))

    preds = torch.ones((10, 2, 10, 10))
    targets = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    preds[:, 1, :, :] = 0
    targets[:, 1, :, :] = 0
    dice_coefficient = dice(preds, targets)
    # Dice for the first label should be 1.0 for all batch components, and the second should be ignored as
    # zero_division is None
    assert dice_coefficient == approx(1.0)

    # change threshold to be 1 and zero division to be 1.0
    dice = MultiClassDice(name="DICE", label_dim=0, zero_division=1.0, threshold=0.1, batch_dim=2)

    targets = torch.ones((10, 2, 10, 10))
    preds = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    preds[:, 1, :, :] = 0
    targets[:, 1, :, :] = 0
    dice_coefficient = dice(preds, targets)
    # Dice for the first label should be 1.0 and the second should be 1.0, since we have all TNs for that channel and
    # zero_division = 1.0
    assert dice_coefficient == approx(1.0)
