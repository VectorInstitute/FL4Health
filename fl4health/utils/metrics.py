from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from flwr.common.typing import Scalar


class Metric(ABC):
    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Scalar:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


class Accuracy(Metric):
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert pred.shape[0] == target.shape[0]
        pred = torch.argmax(pred, 1)
        correct = (pred == target).sum().item()
        accuracy = correct / pred.shape[0]
        return accuracy

    def __str__(self) -> str:
        return "accuracy"


class AverageMeter:
    def __init__(self, metrics: List[Metric], name: str = "") -> None:
        self.metrics: List[Metric] = metrics
        self.name: str = name

        self.metric_values_history: List[List[Scalar]] = [[] for _ in range(len(self.metrics))]
        self.counts: List = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        metric_values: List[Scalar] = [metric(input, target) for metric in self.metrics]
        self.counts.append(target.size(0))

        for i, metric_value in enumerate(metric_values):
            self.metric_values_history[i].append(metric_value)

    def compute(self) -> Dict[str, Scalar]:
        total_count = sum(self.counts)
        weights: List[float] = [count / total_count for count in self.counts]

        metric_value_averages = []
        for metric_values in self.metric_values_history:
            avg = sum([weight * float(val) for weight, val in zip(weights, metric_values)])
            metric_value_averages.append(avg)

        results: Dict[str, Scalar] = {
            f"{self.name}_{str(metric)}": avg for metric, avg in zip(self.metrics, metric_value_averages)
        }

        return results
