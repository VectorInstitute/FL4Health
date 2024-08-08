# type: ignore
import torch
from flwr.common.typing import Scalar
from sklearn import metrics

from fl4health.utils.metrics import Metric


class Binary_ROC_AUC(Metric):
    def __init__(self, name: str = "binary_ROC_AUC"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        # prob = torch.nn.functional.softmax(logits, dim=1)
        prob = logits.cpu().detach()
        target = target.cpu().detach()
        y_true = target.reshape(-1)
        return metrics.roc_auc_score(y_true, prob)


class Binary_F1(Metric):
    def __init__(self, name: str = "binary_F1"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.f1_score(y_true, preds, average="weighted")


class Binary_F1_Macro(Metric):
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


class Binary_Balanced_Accuracy(Metric):
    def __init__(self, name: str = "balanced accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = logits >= 0.5
        return metrics.balanced_accuracy_score(y_true, preds)
