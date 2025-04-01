import math

import torch
from pytest import approx

from fl4health.metrics.efficient_metrics import MultiClassDice


def test_multi_dice_metric_with_threshold() -> None:
    torch.manual_seed(42)

    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    hd = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, threshold=2, zero_division=None)
    hd.update(logits, targets)
    result = hd.compute()
    # Dice scores collapse to D = [[0, 2/(2), 0], [2/(2+1), 2/(2+1), 0]] per instance, class
    assert result["DICE"] == approx((1.0 / 6.0) * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0
    hd.update(logits, targets)
    result = hd.compute()
    # Dice scores collapse to D = [[0, 4/(4+1), 0], [2/(2), 2/(2+1), 2/2]] per instance, class in this batch
    # Adding these to the above and averaging yields
    assert result["DICE"] == approx(
        (1.0 / 12.0)
        * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 2.0 + 2.0 / 3.0 + 2.0 / 2.0)
    )

    # Test Dropping zero divisions
    # Kill the threshold so we can have all zeros for preds
    hd.threshold = None
    logits = torch.zeros((2, 3, 3))
    targets = torch.zeros((2, 3, 3))

    hd.update(logits, targets)
    result = hd.compute()
    # Should be the same since all added pieces are just TNs and are ignored
    assert result["DICE"] == approx(
        (1.0 / 12.0)
        * (0 + 1.0 + 0 + 2.0 / 3.0 + 2.0 / 3.0 + 0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 2.0 + 2.0 / 3.0 + 2.0 / 2.0)
    )

    hd.clear()
    # Test NaN
    logits = torch.zeros((2, 3, 3))
    targets = torch.zeros((2, 3, 3))

    hd.update(logits, targets)
    result = hd.compute()
    assert isinstance(result["DICE"], float)  # Appeases mypy, ensure its a float before checking if its NaN
    assert math.isnan(result["DICE"])


def test_multi_dice_metric_ignore_background() -> None:
    torch.manual_seed(42)

    # Test ignore background
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    hd = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, threshold=2, ignore_background=2, zero_division=None)
    hd.update(logits, targets)
    result = hd.compute()
    # Dice scores collapse to D = [[0, 2/(2), 0], [2/(2+1), 2/(2+1), 0]] per instance, class
    # However, we're dropping the first class using ignore background
    assert result["DICE"] == approx((1.0 / 4.0) * (1.0 + 0 + 2.0 / 3.0 + 0))

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    hd.update(logits, targets)
    result = hd.compute()
    # Dice scores collapse to D = [[0, 4/(4+1), 0], [2/(2), 2/(2+1), 2/2]] per instance, class in this batch
    # However, we're dropping the first class using ignore background
    # Adding these to the above and averaging yields
    assert result["DICE"] == approx((1.0 / 8.0) * (1.0 + 0 + 2.0 / 3.0 + 0 + 4.0 / 5.0 + 0 + 2.0 / 3.0 + 2.0 / 2.0))


def test_continuous_multi_dice_metric() -> None:
    torch.manual_seed(42)

    # Test ignore background
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    hd = MultiClassDice(name="DICE", label_dim=2, batch_dim=0, zero_division=None)
    hd.update(logits, targets)
    result = hd.compute()
    tp1 = [[0, 0.9150, 0.6009], [0.8694 + 0.4294, 0.9346, 0.7411 + 0.5739]]
    fp1 = [[0.8823 + 0.9593 + 0.2566, 0.3904 + 0.7936, 0.3829 + 0.9408], [0.1332, 0.5677 + 0.8854, 0.5936]]
    fn1 = [[0, (1 - 0.9150), (1 - 0.6009)], [(1 - 0.8694) + (1 - 0.4294), (1 - 0.9346), (1 - 0.7411) + (1 - 0.5739)]]
    dice_target_11 = (2 * tp1[0][0]) / (2 * tp1[0][0] + fp1[0][0] + fn1[0][0])
    dice_target_12 = (2 * tp1[0][1]) / (2 * tp1[0][1] + fp1[0][1] + fn1[0][1])
    dice_target_13 = (2 * tp1[0][2]) / (2 * tp1[0][2] + fp1[0][2] + fn1[0][2])
    dice_target_21 = (2 * tp1[1][0]) / (2 * tp1[1][0] + fp1[1][0] + fn1[1][0])
    dice_target_22 = (2 * tp1[1][1]) / (2 * tp1[1][1] + fp1[1][1] + fn1[1][1])
    dice_target_23 = (2 * tp1[1][2]) / (2 * tp1[1][2] + fp1[1][2] + fn1[1][2])
    assert result["DICE"] == approx(
        (1.0 / 6.0)
        * (dice_target_11 + dice_target_12 + dice_target_13 + dice_target_21 + dice_target_22 + dice_target_23),
        abs=1e-4,
    )

    # Test that accumulation works still
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    hd.update(logits, targets)
    result = hd.compute()
    tp2 = [[0.5779 + 0.7104, 0.9040 + 0.6343, 0.5547 + 0.7890], [0.9103, 0.7886 + 0.1165, 0.7539]]
    fp2 = [[0.3423, 0.9464, 0.3644], [0.2814 + 0.3068, 0.1952, 0.5895 + 0.0050]]
    fn2 = [
        [(1 - 0.5779) + (1 - 0.7104), (1 - 0.9040) + (1 - 0.6343), (1 - 0.5547) + (1 - 0.7890)],
        [(1 - 0.9103), (1 - 0.7886) + (1 - 0.1165), (1 - 0.7539)],
    ]
    next_dice_target_11 = (2 * tp2[0][0]) / (2 * tp2[0][0] + fp2[0][0] + fn2[0][0])
    next_dice_target_12 = (2 * tp2[0][1]) / (2 * tp2[0][1] + fp2[0][1] + fn2[0][1])
    next_dice_target_13 = (2 * tp2[0][2]) / (2 * tp2[0][2] + fp2[0][2] + fn2[0][2])
    next_dice_target_21 = (2 * tp2[1][0]) / (2 * tp2[1][0] + fp2[1][0] + fn2[1][0])
    next_dice_target_22 = (2 * tp2[1][1]) / (2 * tp2[1][1] + fp2[1][1] + fn2[1][1])
    next_dice_target_23 = (2 * tp2[1][2]) / (2 * tp2[1][2] + fp2[1][2] + fn2[1][2])
    assert result["DICE"] == approx(
        (1.0 / 12.0)
        * (
            dice_target_11
            + dice_target_12
            + dice_target_13
            + dice_target_21
            + dice_target_22
            + dice_target_23
            + next_dice_target_11
            + next_dice_target_12
            + next_dice_target_13
            + next_dice_target_21
            + next_dice_target_22
            + next_dice_target_23
        ),
        abs=1e-4,
    )


def test_more_spatial_dimensions() -> None:
    metric = MultiClassDice(name="DICE", label_dim=1, batch_dim=2)
    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero
    ones_and_zeros_logits[:, 1, :, :] = 0
    dice_coefficient = metric(ones_and_zeros_logits, all_ones_targets)
    # Dice for the first label should be 1.0 for all batch components and the second should be 0. Thus the final score
    # should be 0.5
    assert dice_coefficient == approx(0.5)

    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero
    ones_and_zeros_logits[:, 1, :, :] = 0
    # Set entries in the second channel to zero for the first 5 instances
    all_ones_targets[:, 1, 0:5, :] = 0
    dice_coefficient = metric(ones_and_zeros_logits, all_ones_targets)
    # Dice for the first label should be 1.0 for all batch components and the second should be 0 for the final five
    # entries and the first 5 entries should be ignored since zero_division = None
    assert dice_coefficient == approx((1.0 / 15.0) * (10 * 1.0 + 5 * 0))

    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    ones_and_zeros_logits[:, 1, :, :] = 0
    all_ones_targets[:, 1, :, :] = 0
    dice_coefficient = metric(ones_and_zeros_logits, all_ones_targets)
    # Dice for the first label should be 1.0 for all batch components, and the second should be ignored as
    # zero_division is None
    assert dice_coefficient == approx(1.0)

    # change threshold to be 1 and zero division to be 1.0
    metric = MultiClassDice(name="DICE", label_dim=0, zero_division=1.0, threshold=0.1, batch_dim=2)
    all_ones_targets = torch.ones((10, 2, 10, 10))
    ones_and_zeros_logits = torch.ones((10, 2, 10, 10))
    # Set entries in the second channel to zero for both target and preds
    ones_and_zeros_logits[:, 1, :, :] = 0
    all_ones_targets[:, 1, :, :] = 0
    dice_coefficient = metric(ones_and_zeros_logits, all_ones_targets)
    # Dice for the first label should be 1.0 and the second should be 1.0, since we have all TNs for that channel
    assert dice_coefficient == approx(1.0)
