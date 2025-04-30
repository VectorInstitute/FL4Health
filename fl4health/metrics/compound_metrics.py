from collections.abc import Sequence
from typing import Generic, TypeVar

import torch
from flwr.common.typing import Metrics

from fl4health.metrics.base_metrics import Metric
from fl4health.utils.typing import TorchTransformFunction

T = TypeVar("T", bound=Metric)


class TransformsMetric(Metric, Generic[T]):
    def __init__(
        self,
        metric: T,
        pred_transforms: Sequence[TorchTransformFunction] | None = None,
        target_transforms: Sequence[TorchTransformFunction] | None = None,
    ) -> None:
        """
        A thin wrapper class to allow transforms to be applied to preds and targets prior to calculating metrics.
        Transforms are applied in the order given

        Args:
            metric (Metric): A FL4Health compatible metric
            pred_transforms (Sequence[TorchTransformFunction] | None, optional): A list of transform functions to
                apply to the model predictions before computing the metrics. Each callable must accept and return a
                ``torch.Tensor``. Use partial to set other arguments.
            target_transforms (Sequence[TorchTransformFunction] | None, optional): A list of transform functions to
                apply to the targets before computing the metrics. Each callable must accept and return a
                ``torch.Tensor``. Use partial to set other arguments.
        """
        self.metric = metric
        self.pred_transforms = [] if pred_transforms is None else pred_transforms
        self.target_transforms = [] if target_transforms is None else target_transforms
        super().__init__(name=self.metric.name)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        for transform in self.pred_transforms:
            pred = transform(pred)

        for transform in self.target_transforms:
            target = transform(target)

        self.metric.update(pred, target)

    def compute(self, name: str | None = None) -> Metrics:
        return self.metric.compute(name)

    def clear(self) -> None:
        return self.metric.clear()
