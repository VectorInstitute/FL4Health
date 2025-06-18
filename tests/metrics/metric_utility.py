import torch
from flwr.common.typing import Scalar
from sklearn import metrics as sklearn_metrics

from fl4health.metrics import SimpleMetric


class AccuracyForTest(SimpleMetric):
    """
    Accuracy class strictly reserved for testing the TransformsMetric. More strongly decouples the test from any
    of our metrics implementations.
    """

    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        preds = (logits > 0.5).int()
        target = target.cpu().detach()
        preds = preds.cpu().detach()
        return sklearn_metrics.accuracy_score(target, preds)
