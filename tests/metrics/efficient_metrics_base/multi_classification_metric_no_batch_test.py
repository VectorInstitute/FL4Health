import re

import pytest
import torch

from fl4health.metrics.efficient_metrics_base import ClassificationOutcome, MultiClassificationMetric
from fl4health.metrics.metrics_utils import threshold_tensor
from fl4health.utils.random import set_all_random_seeds


def test_tensor_thresholding() -> None:
    tensor_to_threshold = torch.Tensor([[1, 2, 3, 4], [6, 7, 1, 2]])  # shape (2, 4)
    float_thresholded = threshold_tensor(tensor_to_threshold, 4.0)
    int_thresholded = threshold_tensor(tensor_to_threshold, 1)

    assert torch.allclose(float_thresholded, torch.Tensor([[0, 0, 0, 0], [1, 1, 0, 0]]))
    assert torch.allclose(int_thresholded, torch.Tensor([[0, 0, 0, 1], [0, 1, 0, 0]]))


def test_remove_background() -> None:
    preds_tensor = torch.Tensor([[[1, 2], [3, 4]], [[6, 7], [1, 2]]])  # shape (2, 2, 2)
    targets_tensor = torch.Tensor([[[2, 2], [3, 5]], [[6, 8], [2, 2]]])  # shape (2, 2, 2)
    preds_with_removal, targets_with_removal = MultiClassificationMetric._remove_background(
        0, preds_tensor, targets_tensor
    )
    # Shape should be (1,2,2) for these tensors equivalent to original[1, :, :]
    assert torch.allclose(preds_with_removal, torch.Tensor([[[6, 7], [1, 2]]]))
    assert torch.allclose(targets_with_removal, torch.Tensor([[[6, 8], [2, 2]]]))

    preds_tensor_to_modify = torch.rand((2, 3, 5, 2))
    target_tensor_to_modify = torch.rand((2, 3, 5, 2))
    preds_with_removal, targets_with_removal = MultiClassificationMetric._remove_background(
        2, preds_tensor_to_modify, target_tensor_to_modify
    )
    assert preds_with_removal.shape == (2, 3, 4, 2)
    assert targets_with_removal.shape == (2, 3, 4, 2)


def test_classification_metric_counts() -> None:
    set_all_random_seeds(42)

    logits = torch.Tensor(
        [
            [3.0, 1.0, 2.0],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 1.1],
            [0.5, 3.0, 1.5],
        ]
    )
    targets = torch.Tensor([0, 0, 2, 0, 2])

    # preds are vector encoded and are to be thresholded and targets are label encoded
    classification_metric = MultiClassificationMetric(name="metric", label_dim=1, threshold=1)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([2, 0, 1]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([2, 4, 2]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([0, 1, 1]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([1, 0, 1]))

    classification_metric.clear()

    # preds are vector encoded and are to be thresholded (by float) and targets are label encoded
    classification_metric = MultiClassificationMetric(name="metric", label_dim=1, threshold=0.5)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([3, 0, 2]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([2, 3, 1]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([0, 2, 2]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([0, 0, 0]))

    classification_metric.clear()

    # Predictions are SOFT and not thresholded. So we get continuous counts
    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = MultiClassificationMetric(name="metric", label_dim=2)
    classification_metric.update(logits, targets)

    assert torch.allclose(
        classification_metric.true_positives,
        torch.Tensor([0.8823 + 0.3829 + 0.3904, 0.9150 + 0.6009 + 0.7936 + 0.5936]),
        atol=1e-4,
    )
    assert torch.allclose(
        classification_metric.true_negatives,
        torch.Tensor([(1 - 0.2566) + (1 - 0.9408) + (1 - 0.9346), (1 - 0.1332) + (1 - 0.9593)]),
        atol=1e-3,
    )
    assert torch.allclose(
        classification_metric.false_positives, torch.Tensor([0.9408 + 0.9346 + 0.2566, 0.9593 + 0.1332]), atol=1e-3
    )
    assert torch.allclose(
        classification_metric.false_negatives,
        torch.Tensor(
            [(1 - 0.8823) + (1 - 0.3829) + (1 - 0.3904), (1 - 0.9150) + (1 - 0.6009) + (1 - 0.7936) + (1 - 0.5936)]
        ),
        atol=1e-4,
    )

    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    tp_target = classification_metric.true_positives + torch.Tensor(
        [0.1053 + 0.3588 + 0.5472 + 0.9516 + 0.8860 + 0.3376, 0.2695 + 0.0753 + 0.8090]
    )
    tn_target = classification_metric.true_negatives + torch.Tensor([0, (1 - 0.1994) + (1 - 0.0062) + (1 - 0.5832)])
    fp_target = classification_metric.false_positives + torch.Tensor([0, 0.1994 + 0.0062 + 0.5832])
    fn_target = classification_metric.false_negatives + torch.Tensor(
        [
            (1 - 0.1053) + (1 - 0.3588) + (1 - 0.5472) + (1 - 0.9516) + (1 - 0.8860) + (1 - 0.3376),
            (1 - 0.2695) + (1 - 0.0753) + (1 - 0.8090),
        ]
    )

    # Accumulate more counts, which should be continuous valued
    classification_metric.update(logits, targets)
    assert torch.allclose(classification_metric.true_positives, tp_target, atol=1e-4)
    assert torch.allclose(classification_metric.true_negatives, tn_target, atol=1e-4)
    assert torch.allclose(classification_metric.false_positives, fp_target, atol=1e-4)
    assert torch.allclose(classification_metric.false_negatives, fn_target, atol=1e-4)

    # Preds are label encoded and targets are not
    logits = torch.argmax(torch.rand((2, 3, 2)), dim=2)
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = MultiClassificationMetric(name="metric", label_dim=2)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([2, 0]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([0, 2]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([1, 3]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([3, 1]))

    # Test that discarding works properly
    logits = torch.Tensor(
        [
            [3.0, 1.0, 2.0],
            [0.88, 0.06, 0.06],
            [0.1, 0.3, 1.2],
            [0.9, 0.3, 1.1],
            [0.5, 3.0, 1.5],
        ]
    )
    targets = torch.Tensor([0, 0, 2, 0, 2])

    # preds are vector encoded and are to be thresholded, targets are label encoded
    classification_metric = MultiClassificationMetric(
        name="metric", label_dim=1, threshold=1, discard={ClassificationOutcome.FALSE_NEGATIVE}
    )
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([2, 0, 1]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([2, 4, 2]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([0, 1, 1]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([]))


def test_appropriate_errors_thrown_when_using_class() -> None:
    binary_or_both_label_index_vectors = re.compile(
        "Label dimension for preds tensor is less than 2. Either your label dimension is a single float value",
        flags=re.IGNORECASE,
    )
    preds_out_of_bounds = re.compile("Expected preds to be in range \\[0, 1\\].", flags=re.IGNORECASE)

    classification_metric = MultiClassificationMetric(name="metric", label_dim=2)

    # Binary setting
    logits = torch.rand((2, 3, 1))
    targets = torch.rand((2, 3, 1))
    with pytest.raises(Exception, match=binary_or_both_label_index_vectors):
        classification_metric.update(logits, targets)

    # Label index encoded setting
    logits = torch.argmax(torch.rand((2, 3, 2)), dim=2).unsqueeze(2)  # shape (2,3,1)
    targets = torch.argmax(torch.rand((2, 3, 2)), dim=2).unsqueeze(2)  # shape (2,3,1)
    with pytest.raises(Exception, match=binary_or_both_label_index_vectors):
        classification_metric.update(logits, targets)

    # Preds/targets are not in [0, 1]
    logits = torch.randn((2, 3, 1))
    targets = torch.randn((2, 3, 1))
    classification_metric = MultiClassificationMetric(name="metric", label_dim=2)
    with pytest.raises(Exception, match=preds_out_of_bounds):
        classification_metric.update(logits, targets)
