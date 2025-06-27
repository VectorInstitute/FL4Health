import re

import pytest
import torch

from fl4health.metrics.efficient_metrics_base import ClassificationOutcome, MultiClassificationMetric
from fl4health.utils.random import set_all_random_seeds


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

    # preds are vector encoded and are to be thresholded, targets are label encoded
    classification_metric = MultiClassificationMetric(name="metric", label_dim=1, batch_dim=0, threshold=1)
    classification_metric.update(logits, targets)

    # Batch dimension is preserved across the counts
    assert torch.allclose(
        classification_metric.true_positives, torch.Tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    )
    assert torch.allclose(
        classification_metric.true_negatives, torch.Tensor([[0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    )
    assert torch.allclose(
        classification_metric.false_positives, torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]])
    )
    assert torch.allclose(
        classification_metric.false_negatives, torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]])
    )

    classification_metric.clear()

    # preds are vector encoded and are to be thresholded (by float), targets are label encoded
    classification_metric = MultiClassificationMetric(name="metric", batch_dim=0, label_dim=1, threshold=0.5)
    classification_metric.update(logits, targets)

    # Batch dimension is preserved across the counts
    assert torch.allclose(
        classification_metric.true_positives, torch.Tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
    )
    assert torch.allclose(
        classification_metric.true_negatives, torch.Tensor([[0, 0, 0], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    )
    assert torch.allclose(
        classification_metric.false_positives, torch.Tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]])
    )
    assert torch.allclose(
        classification_metric.false_negatives, torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    )

    classification_metric.clear()

    # Predictions are SOFT and not thresholded. So we get continuous counts
    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = MultiClassificationMetric(name="metric", batch_dim=0, label_dim=2)
    classification_metric.update(logits, targets)

    assert torch.allclose(
        classification_metric.true_positives,
        torch.Tensor([[0.8823 + 0.3829 + 0.3904, 0.9150 + 0.6009], [0.0, 0.7936 + 0.5936]]),
        atol=1e-4,
    )
    assert torch.allclose(
        classification_metric.true_negatives,
        torch.Tensor([[0.0, (1 - 0.9593)], [(1 - 0.2566) + (1 - 0.9408) + (1 - 0.9346), (1 - 0.1332)]]),
        atol=1e-3,
    )
    assert torch.allclose(
        classification_metric.false_positives,
        torch.Tensor([[0.0, 0.9593], [0.9346 + 0.2566 + 0.9408, 0.1332]]),
        atol=1e-3,
    )
    assert torch.allclose(
        classification_metric.false_negatives,
        torch.Tensor(
            [
                [(1 - 0.8823) + (1 - 0.3829) + (1 - 0.3904), (1 - 0.9150) + (1 - 0.6009)],
                [0.0, (1 - 0.7936) + (1 - 0.5936)],
            ]
        ),
        atol=1e-4,
    )

    logits = torch.rand((2, 3, 2))
    targets = torch.rand((2, 3, 2))
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    tp_target = torch.concat(
        (
            classification_metric.true_positives,
            torch.Tensor([[0.1053 + 0.3588 + 0.5472, 0.2695], [0.9516 + 0.8860 + 0.3376, 0.0753 + 0.8090]]),
        ),
        dim=0,
    )
    tn_target = torch.concat(
        (classification_metric.true_negatives, torch.Tensor([[0, (1 - 0.1994) + (1 - 0.0062)], [0, (1 - 0.5832)]])),
        dim=0,
    )
    fp_target = torch.concat(
        (classification_metric.false_positives, torch.Tensor([[0, 0.1994 + 0.0062], [0, 0.5832]])), dim=0
    )
    fn_target = torch.concat(
        (
            classification_metric.false_negatives,
            torch.Tensor(
                [
                    [(1 - 0.1053) + (1 - 0.3588) + (1 - 0.5472), (1 - 0.2695)],
                    [(1 - 0.9516) + (1 - 0.8860) + (1 - 0.3376), (1 - 0.0753) + (1 - 0.8090)],
                ]
            ),
        ),
        dim=0,
    )

    # Accumulate more counts, predictions remain soft, so continuous counts are expected
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

    classification_metric = MultiClassificationMetric(name="metric", batch_dim=0, label_dim=2)
    classification_metric.update(logits, targets)

    assert torch.allclose(classification_metric.true_positives, torch.Tensor([[1, 0], [1, 0]]))
    assert torch.allclose(classification_metric.true_negatives, torch.Tensor([[0, 1], [0, 1]]))
    assert torch.allclose(classification_metric.false_positives, torch.Tensor([[0, 2], [1, 1]]))
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([[2, 0], [1, 1]]))

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
    # preds are vector encoded and need to be thresholded and targets are label encoded
    classification_metric = MultiClassificationMetric(
        name="metric", batch_dim=0, label_dim=1, threshold=1, discard={ClassificationOutcome.FALSE_NEGATIVE}
    )
    classification_metric.update(logits, targets)

    assert torch.allclose(
        classification_metric.true_positives, torch.Tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]])
    )
    assert torch.allclose(
        classification_metric.true_negatives, torch.Tensor([[0, 1, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    )
    assert torch.allclose(
        classification_metric.false_positives, torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]])
    )
    assert torch.allclose(classification_metric.false_negatives, torch.Tensor([]))

    # Test when label dim is less than batch dim that everything still comes out properly
    set_all_random_seeds(42)
    # Need to rearrange the tensors to maintain the meaning of labels and batch dim while having batch dim come after
    logits = torch.rand((2, 3, 2)).transpose(0, 1).transpose(1, 2)
    targets = torch.rand((2, 3, 2)).transpose(0, 1).transpose(1, 2)
    mask_1 = targets > 0.5
    targets = torch.zeros_like(targets)
    targets[mask_1] = 1.0

    classification_metric = MultiClassificationMetric(name="metric", batch_dim=2, label_dim=1)
    classification_metric.update(logits, targets)

    tp_target = torch.Tensor([[0.9150 + 0.6009], [0.7936 + 0.5936]])
    tn_target = torch.Tensor([[1 - 0.9593], [1 - 0.1332]])
    fp_target = torch.Tensor([[0.9593], [0.1332]])
    fn_target = torch.Tensor([[(1 - 0.9150) + (1 - 0.6009)], [(1 - 0.7936) + (1 - 0.5936)]])

    assert torch.allclose(
        classification_metric.true_positives,
        torch.Tensor([[0.8823 + 0.3829 + 0.3904, 0.9150 + 0.6009], [0.0, 0.7936 + 0.5936]]),
        atol=1e-4,
    )
    assert torch.allclose(
        classification_metric.true_negatives,
        torch.Tensor([[0.0, (1 - 0.9593)], [(1 - 0.2566) + (1 - 0.9408) + (1 - 0.9346), (1 - 0.1332)]]),
        atol=1e-3,
    )
    assert torch.allclose(
        classification_metric.false_positives,
        torch.Tensor([[0.0, 0.9593], [0.9346 + 0.2566 + 0.9408, 0.1332]]),
        atol=1e-3,
    )
    assert torch.allclose(
        classification_metric.false_negatives,
        torch.Tensor(
            [
                [(1 - 0.8823) + (1 - 0.3829) + (1 - 0.3904), (1 - 0.9150) + (1 - 0.6009)],
                [0.0, (1 - 0.7936) + (1 - 0.5936)],
            ]
        ),
        atol=1e-4,
    )


def test_appropriate_errors_thrown_when_using_class() -> None:
    binary_or_both_label_index_vectors = re.compile(
        "Label dimension for preds tensor is less than 2. Either your label dimension is a single float value",
        flags=re.IGNORECASE,
    )
    preds_out_of_bounds = re.compile("Expected preds to be in range \\[0, 1\\].", flags=re.IGNORECASE)

    classification_metric = MultiClassificationMetric(name="metric", batch_dim=0, label_dim=2)

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
    classification_metric = MultiClassificationMetric(name="metric", batch_dim=0, label_dim=2)
    with pytest.raises(Exception, match=preds_out_of_bounds):
        classification_metric.update(logits, targets)
