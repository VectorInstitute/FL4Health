import re

import pytest
import torch

from fl4health.metrics.efficient_metrics_base import BinaryClassificationMetric, ClassificationOutcome
from fl4health.utils.random import set_all_random_seeds


def test_binary_classification_metric_counts() -> None:
    set_all_random_seeds(42)

    logits = torch.Tensor([[3.0, 1.0], [0.88, 0.06], [0.1, 0.3], [0.9, 0.3], [0.5, 3.0]])
    targets = torch.Tensor([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]])

    # preds are vector encoded and are to be thresholded and targets are also vector encoded
    classification_metric = BinaryClassificationMetric(name="metric", label_dim=1, batch_dim=0, threshold=1)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([[0], [0], [1], [0], [1]]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([[1], [1], [0], [1], [0]]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[0], [0], [0], [0], [0]]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[0], [0], [0], [0], [0]]))

    # Get stats for negative label instead
    classification_metric = BinaryClassificationMetric(
        name="metric", label_dim=1, batch_dim=0, threshold=1, pos_label=0
    )
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([[0], [0], [1], [0], [1]]))
    assert torch.allclose(classification_metric.true_positives, torch.Tensor([[1], [1], [0], [1], [0]]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[0], [0], [0], [0], [0]]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[0], [0], [0], [0], [0]]))

    # preds are vector encoded and are to be thresholded (by float) and targets are label encoded
    classification_metric = BinaryClassificationMetric(name="metric", label_dim=1, batch_dim=0, threshold=0.5)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([[0], [0], [0], [0], [1]]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([[0], [1], [0], [1], [0]]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[1], [0], [0], [0], [0]]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[0], [0], [1], [0], [0]]))

    classification_metric.clear()

    # Predictions are SOFT and not {0, 1}. So we get continuous counts
    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = BinaryClassificationMetric(name="metric", batch_dim=0, label_dim=2)
    classification_metric.update(logits, targets)

    tp_target = torch.Tensor([[0.9150 + 0.6009], [0.7936 + 0.5936]])
    tn_target = torch.Tensor([[1 - 0.9593], [1 - 0.1332]])
    fp_target = torch.Tensor([[0.9593], [0.1332]])
    fn_target = torch.Tensor([[(1 - 0.9150) + (1 - 0.6009)], [(1 - 0.7936) + (1 - 0.5936)]])

    assert torch.allclose(classification_metric.true_positives, tp_target, atol=1e-4)
    assert torch.allclose(classification_metric.true_negatives, tn_target, atol=1e-3)
    assert torch.allclose(classification_metric.false_positives, fp_target, atol=1e-3)
    assert torch.allclose(classification_metric.false_negatives, fn_target, atol=1e-4)

    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    tp_target = torch.Tensor([[0.9150 + 0.6009], [0.7936 + 0.5936], [0.2695], [0.0753 + 0.8090]])
    tn_target = torch.Tensor([[1 - 0.9593], [1 - 0.1332], [(1 - 0.1994) + (1 - 0.0062)], [1 - 0.5832]])
    fp_target = torch.Tensor([[0.9593], [0.1332], [0.1994 + 0.0062], [0.5832]])
    fn_target = torch.Tensor(
        [[(1 - 0.9150) + (1 - 0.6009)], [(1 - 0.7936) + (1 - 0.5936)], [1 - 0.2695], [(1 - 0.0753) + (1 - 0.8090)]]
    )

    # Accumulate more counts, which should be continuous valued
    classification_metric.update(logits, targets)
    assert torch.allclose(classification_metric.true_positives, tp_target, atol=1e-4)
    assert torch.allclose(classification_metric.true_negatives, tn_target, atol=1e-4)
    assert torch.allclose(classification_metric.false_positives, fp_target, atol=1e-4)
    assert torch.allclose(classification_metric.false_negatives, fn_target, atol=1e-4)

    # Preds are continuous values, both preds and targets implicitly encoded (i.e. not vector encoded)
    logits = torch.rand((2, 3))
    targets = torch.rand((2, 3))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = BinaryClassificationMetric(name="metric", batch_dim=0)
    classification_metric.update(logits, targets)

    assert torch.allclose(
        classification_metric.true_positives, torch.Tensor([[0.7539 + 0.1952 + 0.0050], [0.1165]]), atol=1e-3
    )
    assert torch.allclose(
        classification_metric.true_negatives, torch.Tensor([[0.0], [(1 - 0.3068) + (1 - 0.9103)]]), atol=1e-4
    )
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[0.0], [0.3068 + 0.9103]]), atol=1e-4)
    assert torch.allclose(
        classification_metric.false_negatives,
        torch.Tensor([[(1 - 0.7539) + (1 - 0.1952) + (1 - 0.0050)], [1 - 0.1165]]),
        atol=1e-4,
    )

    # Change the pos_label to zero and make sure everything is rearranged properly
    classification_metric = BinaryClassificationMetric(name="metric", pos_label=0, batch_dim=0)
    classification_metric.update(logits, targets)

    assert torch.allclose(
        classification_metric.true_negatives, torch.Tensor([[0.7539 + 0.1952 + 0.0050], [0.1165]]), atol=1e-3
    )
    assert torch.allclose(
        classification_metric.true_positives, torch.Tensor([[0.0], [(1 - 0.3068) + (1 - 0.9103)]]), atol=1e-4
    )
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[0.0], [0.3068 + 0.9103]]), atol=1e-4)
    assert torch.allclose(
        classification_metric.false_positives,
        torch.Tensor([[(1 - 0.7539) + (1 - 0.1952) + (1 - 0.0050)], [1 - 0.1165]]),
        atol=1e-4,
    )

    # Test that discarding happens properly
    logits = torch.Tensor([[3.0, 1.0], [0.88, 0.06], [0.1, 0.3], [0.9, 0.3], [0.5, 3.0]])
    targets = torch.Tensor([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]])

    # preds are vector encoded and need to be thresholded and targets are also vector encoded
    classification_metric = BinaryClassificationMetric(
        name="metric",
        batch_dim=0,
        label_dim=1,
        threshold=1,
        discard={ClassificationOutcome.TRUE_POSITIVE, ClassificationOutcome.TRUE_NEGATIVE},
    )
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[0], [0], [0], [0], [0]]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[0], [0], [0], [0], [0]]))

    # Test when label dim is less than batch dim that everything still comes out properly
    set_all_random_seeds(42)
    # Need to rearrange the tensors to maintain the meaning of labels and batch dim while having batch dim come after
    logits = torch.rand((2, 3, 2)).transpose(0, 1).transpose(1, 2)
    targets = torch.rand((2, 3, 2)).transpose(0, 1).transpose(1, 2)
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = BinaryClassificationMetric(name="metric", batch_dim=2, label_dim=1)
    classification_metric.update(logits, targets)

    tp_target = torch.Tensor([[0.9150 + 0.6009], [0.7936 + 0.5936]])
    tn_target = torch.Tensor([[1 - 0.9593], [1 - 0.1332]])
    fp_target = torch.Tensor([[0.9593], [0.1332]])
    fn_target = torch.Tensor([[(1 - 0.9150) + (1 - 0.6009)], [(1 - 0.7936) + (1 - 0.5936)]])

    assert torch.allclose(classification_metric.true_positives, tp_target, atol=1e-4)
    assert torch.allclose(classification_metric.true_negatives, tn_target, atol=1e-3)
    assert torch.allclose(classification_metric.false_positives, fp_target, atol=1e-3)
    assert torch.allclose(classification_metric.false_negatives, fn_target, atol=1e-4)


def test_appropriate_errors_thrown_when_using_class() -> None:
    binary_or_both_label_index_vectors = re.compile(
        "Label dimension for preds tensor is greater than 2", flags=re.IGNORECASE
    )
    preds_out_of_bounds = re.compile("Expected preds to be in range \\[0, 1\\].", flags=re.IGNORECASE)
    bad_pos_label_value = re.compile("pos_label must be either 0 or 1", flags=re.IGNORECASE)
    preds_and_targets_different_shapes = re.compile("Preds and targets must have the same shape", flags=re.IGNORECASE)

    # Multi-class setting (binary class not valid)
    logits = torch.rand((2, 3, 3))
    targets = torch.rand((2, 3, 3))
    classification_metric = BinaryClassificationMetric(name="metric", label_dim=2, batch_dim=0)
    with pytest.raises(Exception, match=binary_or_both_label_index_vectors):
        classification_metric.update(logits, targets)

    # Bad pos_label provided
    with pytest.raises(Exception, match=bad_pos_label_value):
        classification_metric = BinaryClassificationMetric(name="metric", pos_label=2, batch_dim=0)

    # Preds/Targets out of bounds
    logits = torch.randn((2, 3, 1))
    targets = torch.randn((2, 3, 1))
    classification_metric = BinaryClassificationMetric(name="metric", label_dim=2, batch_dim=0)
    with pytest.raises(Exception, match=preds_out_of_bounds):
        classification_metric.update(logits, targets)

    # Preds/targets are not of the same shape
    logits = torch.rand((2, 3))
    targets = torch.rand((2, 3, 1))
    classification_metric = BinaryClassificationMetric(name="metric", batch_dim=0)
    with pytest.raises(Exception, match=preds_and_targets_different_shapes):
        classification_metric.update(logits, targets)
