from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from flwr.common.typing import Scalar


class Metric(ABC):
    """
    Abstact class to be extended to create metric functions used to evaluate the
    predictions of a model.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Scalar:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class Accuracy(Metric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert pred.shape[0] == target.shape[0]
        pred = torch.argmax(pred, 1)
        correct = (pred == target).sum().item()
        accuracy = correct / pred.shape[0]
        return accuracy


class AverageMeter:
    """
    class used to compute the average of metrics iteratively evaluated over a set of prediction-target pairings.
    The constructor takes a list of type Metric. These metrics are then evaluated each time the update method is
    called with predcitions and ground truth labels. The count corresponding to each evaluation is stored to ensure
    the metrics average is accurate. The compute method is used to return a dictionairy of metrics along with their
    current values.
    """

    def __init__(self, metrics: List[Metric], name: str = "") -> None:
        self.metrics: List[Metric] = metrics
        self.name: str = name

        self.metric_values_history: List[List[Scalar]] = [[] for _ in range(len(self.metrics))]
        self.counts: List = []

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
