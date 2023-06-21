from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np
import torch
from flwr.common.typing import Scalar
from sklearn import metrics


class Metric(ABC):
    """
    Abstact class to be extended to create metric functions used to evaluate the
    predictions of a model.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class Accuracy(Metric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        preds = torch.argmax(logits, 1)
        correct = (preds == target).sum().item()
        accuracy = correct / preds.shape[0]
        return accuracy


class BalancedAccuracy(Metric):
    def __init__(self, name: str = "balanced_accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = np.argmax(logits, axis=1)
        return metrics.balanced_accuracy_score(y_true, preds)


class ROC_AUC(Metric):
    def __init__(self, name: str = "ROC_AUC score"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        prob = torch.nn.functional.softmax(logits, dim=1)
        prob = prob.cpu().detach()
        target = target.cpu().detach()
        y_true = target.reshape(-1)
        return metrics.roc_auc_score(y_true, prob, average="weighted", multi_class="ovr")


class F1(Metric):
    def __init__(self, name: str = "F1 score"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = np.argmax(logits, axis=1)
        return metrics.f1_score(y_true, preds, average="weighted")


class Meter(ABC):
    def __init__(self, metrics: Sequence[Metric], name: str = "") -> None:
        self.metrics: Sequence[Metric] = metrics
        self.name: str = name

    @abstractmethod
    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        # Update the meter with batch input and target values
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, Scalar]:
        # Compute final metric representations based on the underlying metrics provided to the meter
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class AccumulationMeter(Meter):
    """
    This meter class is used to for metrics that require accumulation of input and target values. That is, they are not
    compatible with computing via weighted averages.
    """

    def __init__(self, metrics: Sequence[Metric], name: str = "") -> None:
        super().__init__(metrics, name)
        self.accumulated_inputs: List[torch.Tensor] = []
        self.accumulated_targets: List[torch.Tensor] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        self.accumulated_inputs.append(input)
        self.accumulated_targets.append(target)

    def compute(self) -> Dict[str, Scalar]:
        metric_values = []
        stacked_inputs = torch.cat(self.accumulated_inputs)
        stacked_targets = torch.cat(self.accumulated_targets)
        for metric in self.metrics:
            metric_values.append(metric(stacked_inputs, stacked_targets))

        results: Dict[str, Scalar] = {
            f"{self.name}_{str(metric)}".lstrip("_"): metric_value
            for metric, metric_value in zip(self.metrics, metric_values)
        }

        return results

    def clear(self) -> None:
        self.accumulated_inputs = []
        self.accumulated_targets = []


class AverageMeter(Meter):
    """
    class used to compute the average of metrics iteratively evaluated over a set of prediction-target pairings.
    The constructor takes a list of type Metric. These metrics are then evaluated each time the update method is
    called with predcitions and ground truth labels. The count corresponding to each evaluation is stored to ensure
    the metrics average is accurate. The compute method is used to return a dictionairy of metrics along with their
    current values.
    """

    def __init__(self, metrics: Sequence[Metric], name: str = "") -> None:
        super().__init__(metrics, name)
        self.metric_values_history: List[List[Scalar]] = [[] for _ in range(len(self.metrics))]
        self.counts: List[int] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        Evaluate metrics and store results.
        """
        metric_values: List[Scalar] = [metric(input, target) for metric in self.metrics]
        self.counts.append(target.size(0))

        for i, metric_value in enumerate(metric_values):
            self.metric_values_history[i].append(metric_value)

    def compute(self) -> Dict[str, Scalar]:
        """
        Returns average of each metrics given its historical values and counts
        """
        total_count = sum(self.counts)
        weights: List[float] = [count / total_count for count in self.counts]

        metric_value_averages = []
        for metric_values in self.metric_values_history:
            avg = sum([weight * float(val) for weight, val in zip(weights, metric_values)])
            metric_value_averages.append(avg)

        results: Dict[str, Scalar] = {
            f"{self.name}_{str(metric)}".lstrip("_"): avg for metric, avg in zip(self.metrics, metric_value_averages)
        }

        return results

    def clear(self) -> None:
        self.metric_values_history = [[] for _ in range(len(self.metrics))]
        self.counts = []
