# type: ignore
import torch
from flwr.common.typing import Scalar
from sklearn import metrics

from fl4health.metrics.base_metrics import Metric


class BinaryRocAuc(Metric):
    def __init__(self, name: str = "binary_ROC_AUC"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        # prob = torch.nn.functional.softmax(logits, dim=1)
        prob = logits.cpu().detach()
        target = target.cpu().detach()
        y_true = target.reshape(-1)
        return metrics.roc_auc_score(y_true, prob)


class BinaryF1(Metric):
    def __init__(self, name: str = "binary_F1"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.f1_score(y_true, preds, average="weighted")


class BinaryF1Macro(Metric):
    def __init__(self, name: str = "binary_F1_macro"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.f1_score(y_true, preds, average="macro")


class Accuracy(Metric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.accuracy_score(y_true, preds)


class BinaryBalancedAccuracy(Metric):
    def __init__(self, name: str = "balanced accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.balanced_accuracy_score(y_true, preds)
