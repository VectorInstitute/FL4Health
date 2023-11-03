import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from flwr.common.typing import Metrics, Optional, Scalar
from sklearn import metrics


class Metric(ABC):
    @abstractmethod
    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self, name: Optional[str]) -> Metrics:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class SimpleMetric(Metric):
    """
    Abstract class to be extended to create metric functions used to evaluate the
    predictions of a model.
    """

    def __init__(self, name: str):
        self.name = name
        self.accumulated_inputs: List[torch.Tensor] = []
        self.accumulated_targets: List[torch.Tensor] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        self.accumulated_inputs.append(input)
        self.accumulated_targets.append(target)

    def compute(self, name: Optional[str] = None) -> Metrics:
        stacked_inputs = torch.cat(self.accumulated_inputs)
        stacked_targets = torch.cat(self.accumulated_targets)
        result = self.__call__(stacked_inputs, stacked_targets)
        result_key = f"{name} - {self.name}" if name is not None else self.name

        return {result_key: result}

    def __str__(self) -> str:
        return self.name

    def clear(self) -> None:
        self.accumulated_inputs = []
        self.accumulated_targets = []

    @abstractmethod
    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        raise NotImplementedError


class BinarySoftDiceCoefficient(SimpleMetric):
    def __init__(
        self,
        name: str = "BinarySoftDiceCoefficient",
        epsilon: float = 1.0e-7,
        spatial_dimensions: Tuple[int, ...] = (2, 3, 4),
        logits_threshold: Optional[float] = 0.5,
    ):
        # Correction term on the DICE denominator calculation
        self.epsilon = epsilon
        # The spatial dimensions of the image within the prediction tensors. The default assumes that the images are 3D
        # and have shape batch_size, channel, spatial, spatial, spatial
        self.spatial_dimensions = spatial_dimensions
        # This is a threshold value where values above are classified as 1 and those below are mapped to 0. If the
        # threshod is None, then no thresholding is performed and a continuous or "soft" DICE coeff. is computed
        self.logits_threshold = logits_threshold
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # Assuming the logits are to be mapped to binary. Note that this assumes the logits have already been
        # constrained to [0, 1]. The metric still functions if not, but results will be unpredictable.
        if self.logits_threshold:
            y_pred = (logits > self.logits_threshold).int()
        else:
            y_pred = logits
        intersection = (y_pred * target).sum(dim=self.spatial_dimensions)
        union = (0.5 * (y_pred + target)).sum(dim=self.spatial_dimensions)
        dice = intersection / (union + self.epsilon)
        # If both inputs are empty the dice coefficient should be equal 1
        dice[union == 0] = 1
        return torch.mean(dice).item()


class Accuracy(SimpleMetric):
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        # Single value output, assume binary logits
        if len(logits.shape) == 1 or logits.shape[1] == 1:
            preds = (logits > 0.5).int()
        else:
            preds = torch.argmax(logits, 1)
        target = target.cpu().detach()
        preds = preds.cpu().detach()
        return metrics.accuracy_score(target, preds)


class BalancedAccuracy(SimpleMetric):
    def __init__(self, name: str = "balanced_accuracy"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = np.argmax(logits, axis=1)
        return metrics.balanced_accuracy_score(y_true, preds)


class ROC_AUC(SimpleMetric):
    def __init__(self, name: str = "ROC_AUC score"):
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        prob = torch.nn.functional.softmax(logits, dim=1)
        prob = prob.cpu().detach()
        target = target.cpu().detach()
        y_true = target.reshape(-1)
        return metrics.roc_auc_score(y_true, prob, average="weighted", multi_class="ovr")


class F1(SimpleMetric):
    def __init__(
        self,
        name: str = "F1 score",
        average: Optional[str] = "weighted",
    ):
        """
        Computes the F1 score using the sklearn f1_score function. As such, the values of average are correspond to
        those of that function.

        Args:
            name (str, optional): Name of the metric. Defaults to "F1 score".
            average (Optional[str], optional): Whether to perform averaging of the F1 scores and how. The values of
                this string corresponds to those of the sklearn f1_score function. See:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
                Defaults to "weighted".
        """
        super().__init__(name)
        self.average = average

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = np.argmax(logits, axis=1)
        return metrics.f1_score(y_true, preds, average=self.average)


class MetricManager:
    """
    Class to manage manage a set of metrics associated to a given prediction type.
    """

    def __init__(self, metrics: Sequence[Metric], meter_mngr_name: str) -> None:
        self.og_metrics = metrics
        self.meter_mngr_name = meter_mngr_name
        self.metrics_per_prediction_type: Dict[str, Sequence[Metric]] = {}

    def update(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        if len(self.metrics_per_prediction_type) == 0:
            self.metrics_per_prediction_type = {key: copy.deepcopy(self.og_metrics) for key in preds.keys()}

        for pred, mtrcs in zip(preds.values(), self.metrics_per_prediction_type.values()):
            for m in mtrcs:
                m.update(pred, target)

    def compute(self) -> Metrics:
        all_results = {}
        for metrics_key, mtrcs in self.metrics_per_prediction_type.items():
            for m in mtrcs:
                result = m.compute(f"{self.meter_mngr_name} - {metrics_key}")
                all_results.update(result)

        return all_results

    def clear(self) -> None:
        self.metrics_per_prediction_type = {}
