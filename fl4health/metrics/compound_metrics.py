import copy
from collections.abc import Sequence
from logging import WARNING
from typing import Generic, TypeVar

import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics

from fl4health.metrics.base_metrics import Metric
from fl4health.utils.typing import TorchTransformFunction


T = TypeVar("T", bound=Metric)


class EmaMetric(Metric, Generic[T]):
    def __init__(self, metric: T, smoothing_factor: float = 0.1, name: str | None = None):
        """
        Exponential Moving Average (EMA) metric wrapper to apply EMA to the underlying metric.

        **NOTE**: If the underlying metric accumulates batches during update, then updating this metric without
        clearing in between will result in previously seen inputs and targets being a part of subsequent computations.
        For example, if we use ``Accuracy`` from ``fl4health.metrics``, which accumulates batches, we get the following
        behavior in the code block below.

        ```python
        from fl4health.metrics import Accuracy

        ema = EmaMetric(Accuracy(), 0.1)

        preds_1 = torch.Tensor([1, 0, 1]), targets_1 = torch.Tensor([1, 1, 1])

        ema.update(preds_1, targets_1)

        ema.compute() -> 0.667

        preds_2 = torch.Tensor([0, 0, 1]), targets_2 = torch.Tensor([1, 1, 1])

        # If no clear before update (new accuracy is computed using both pred_1 and pred_2)

        ema.update(preds_2, targets_2) = 0.9(0.667) + 0.1 (0.5)

        # If there were a clear before update (new accuracy is computed using pred_2)

        ema.clear()

        ema.update(preds_2, targets_2 = 0.9(0.667) + 0.1(0.333)
        ```

        Args:
            metric (T): An FL4Health compatible metric.
            smoothing_factor (float, optional): Smoothing factor in range [0, 1] for the EMA. Smaller values increase
                smoothing by weighting previous scores more heavily. Defaults to 0.1.
            name (str | None, optional): Name of the ``EMAMetric``. If left as None will default to
                'EMA_{metric.name}'.

        """
        # Create a copy of the metrics object so that we do not inadvertently change the provided object elsewhere
        self.metric = copy.deepcopy(metric)
        assert 0.0 <= smoothing_factor <= 1.0, f"smoothing_factor should be in [0, 1] but was {smoothing_factor}"
        self.smoothing_factor = smoothing_factor
        self.previous_score: Metrics | None = None
        self.name = f"EMA_{self.metric.name}" if name is None else name

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        return self.metric.update(input, target)

    def compute(self, name: str | None = None) -> Metrics:
        """
        Compute metric on state accumulated over updates.

        This computation considers the exponential moving average with respect to previous scores. For time step
        \\(t\\), and metric score \\(m_t\\), the EMA score is computed

        \\[
        \\text{smoothing_factor} \\cdot m_t + (1-\\text{smoothing_factor}) \\cdot (m_{t-1}).
        \\]

        The very first score is stored as is.

        Args:
            name (str | None, optional): Optional name used in conjunction with class attribute name to define key in
                metrics dictionary. Defaults to None.

        Returns:
            (Metrics): A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
        """
        # Temporarily change name of the underlying metric so that we get the EMAMetric name in keys of metrics_dict
        metric_name = self.metric.name
        self.metric.name = self.name
        metrics_dict = self.metric.compute(name)
        self.metric.name = metric_name

        # Check if this is the first score
        if self.previous_score is None:
            self._drop_str_or_bytes_scores_and_store(metrics_dict)
            assert self.previous_score is not None
            return copy.deepcopy(self.previous_score)

        # Otherwise compute EMA score for each 'metric' in Metrics dict
        for key, previous_score in self.previous_score.items():
            current_score = metrics_dict[key]
            if not isinstance(current_score, (str, bytes)) and not isinstance(previous_score, (str, bytes)):
                self.previous_score[key] = (
                    self.smoothing_factor * current_score + (1 - self.smoothing_factor) * previous_score
                )

        return copy.deepcopy(self.previous_score)

    def _drop_str_or_bytes_scores_and_store(self, metrics_dict: Metrics) -> None:
        self.previous_score = {}
        for key, score in metrics_dict.items():
            if not isinstance(score, (int, float)):
                log(
                    WARNING,
                    "EMAMetric is only compatible with float or int metrics, but metrics contains a value with "
                    f"type: {type(score)} at key: {key}. These values will be ignored in subsequent computations.",
                )
            else:
                self.previous_score[key] = score

    def clear(self) -> None:
        # Clear accumulated inputs and targets but not the previous score
        return self.metric.clear()


class TransformsMetric(Metric, Generic[T]):
    def __init__(
        self,
        metric: T,
        pred_transforms: Sequence[TorchTransformFunction] | None = None,
        target_transforms: Sequence[TorchTransformFunction] | None = None,
    ) -> None:
        """
        A thin wrapper class to allow transforms to be applied to preds and targets prior to calculating metrics.
        Transforms are applied in the order given.

        Args:
            metric (Metric): A FL4Health compatible metric
            pred_transforms (Sequence[TorchTransformFunction] | None, optional): A list of transform functions to
                apply to the model predictions before computing the metrics. Each callable must accept and return a
                ``torch.Tensor``. Use partial to set other arguments.
            target_transforms (Sequence[TorchTransformFunction] | None, optional): A list of transform functions to
                apply to the targets before computing the metrics. Each callable must accept and return a
                ``torch.Tensor``. Use partial to set other arguments.
        """
        # Create a copy of the metrics object so that we do not inadvertently change the provided object elsewhere
        self.metric = copy.deepcopy(metric)
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
