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
        pred = torch.max(pred, 1)[1]
        correct = (pred == target).sum().item()
        accuracy = correct / pred.shape[0]
        return accuracy

    def __str__(self) -> str:
        return "accuracy"


class AverageMeter:
    def __init__(self, metrics: List[Metric], name: str = "") -> None:
        self.metrics: List[Metric] = metrics
        self.name: str = name

        self.vals: List[List] = [[] for _ in range(len(self.metrics))]
        self.counts: List = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        metric_vals: List[Scalar] = [metric(input, target) for metric in self.metrics]
        self.counts.append(target.size(0))

        for i, metric_val in enumerate(metric_vals):
            self.vals[i].append(metric_val)

    def compute(self) -> Dict[str, Scalar]:
        weights: List[float] = [count / sum(self.counts) for count in self.counts]

        avgs = []
        for lst in self.vals:
            avg = sum([weight * val for weight, val in zip(weights, lst)])
            avgs.append(avg)

        results: Dict[str, Scalar] = {f"{self.name}_{str(metric)}": avg for metric, avg in zip(self.metrics, avgs)}

        return results
