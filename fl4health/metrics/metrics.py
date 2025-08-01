from abc import ABC, abstractmethod

import numpy as np
import torch
from flwr.common.typing import Metrics, Scalar
from sklearn import metrics as sklearn_metrics
from torchmetrics import Metric as TMetric

from fl4health.metrics.base_metrics import Metric


class TorchMetric(Metric):
    def __init__(self, name: str, metric: TMetric) -> None:
        """
        Thin wrapper on ``TorchMetric`` to make it compatible with our ``Metric`` interface.

        Args:
            name (str): The name of the metric.
            metric (TMetric): ``TorchMetric`` class based metric.
        """
        super().__init__(name)
        self.metric = metric

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the state of the underlying ``TorchMetric``.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.
        """
        self.metric.update(input, target.long())

    def compute(self, name: str | None = None) -> Metrics:
        """
        Compute value of underlying ``TorchMetric``.

        Args:
            name (str | None): Optional name used in conjunction with class attribute name to define key in metrics
                dictionary.

        Returns:
           Metrics: A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
        """
        result_key = f"{name} - {self.name}" if name is not None else self.name
        result = self.metric.compute().item()
        return {result_key: result}

    def clear(self) -> None:
        self.metric.reset()


class SimpleMetric(Metric, ABC):
    def __init__(self, name: str) -> None:
        """
        Abstract metric class with base functionality to update, compute and clear metrics. User needs to define
        ``__call__`` method which returns metric given inputs and target.

        Args:
            name (str): Name of the metric.
        """
        super().__init__(name)
        self.accumulated_inputs: list[torch.Tensor] = []
        self.accumulated_targets: list[torch.Tensor] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        This method updates the state of the metric by appending the passed input and target pairing to their
        respective list.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.
        """
        self.accumulated_inputs.append(input)
        self.accumulated_targets.append(target)

    def compute(self, name: str | None = None) -> Metrics:
        """
        Compute metric on accumulated input and output over updates.

        Args:
            name (str | None): Optional name used in conjunction with class attribute name to define key in metrics
                dictionary.

        Raises:
            AssertionError: Input and target lists must be non empty.

        Returns:
            Metrics: A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
        """
        assert len(self.accumulated_inputs) > 0 and len(self.accumulated_targets) > 0
        stacked_inputs = torch.cat(self.accumulated_inputs)
        stacked_targets = torch.cat(self.accumulated_targets)
        result = self.__call__(stacked_inputs, stacked_targets)
        result_key = f"{name} - {self.name}" if name is not None else self.name

        return {result_key: result}

    def clear(self) -> None:
        """Resets metrics by clearing input and target lists."""
        self.accumulated_inputs = []
        self.accumulated_targets = []

    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        """
        User defined method that calculates the desired metric given the predictions and target.

        Raises:
            NotImplementedError: User must define this method.
        """
        raise NotImplementedError


class BinarySoftDiceCoefficient(SimpleMetric):
    def __init__(
        self,
        name: str = "BinarySoftDiceCoefficient",
        epsilon: float = 1.0e-7,
        spatial_dimensions: tuple[int, ...] = (2, 3, 4),
        logits_threshold: float | None = 0.5,
    ):
        """
        Binary DICE Coefficient Metric with configurable spatial dimensions and logits threshold.

        Args:
            name (str): Name of the metric.
            epsilon (float): Small float to add to denominator of DICE calculation to avoid divide by 0.
            spatial_dimensions (tuple[int, ...]): The spatial dimensions of the image within the prediction tensors.
                The default assumes that the images are 3D and have shape:
                (``batch_size``, ``channel``, ``spatial``, ``spatial``, ``spatial``)
            logits_threshold: This is a threshold value where values above are classified as 1 and those below are
                mapped to 0. If the threshold is None, then no thresholding is performed and a continuous or "soft"
                DICE coefficient is computed.
        """
        self.epsilon = epsilon
        self.spatial_dimensions = spatial_dimensions

        self.logits_threshold = logits_threshold
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # Assuming the logits are to be mapped to binary. Note that this assumes the logits have already been
        # constrained to [0, 1]. The metric still functions if not, but results will be unpredictable.
        y_pred = (logits > self.logits_threshold).int() if self.logits_threshold else logits
        intersection = (y_pred * target).sum(dim=self.spatial_dimensions)
        union = (0.5 * (y_pred + target)).sum(dim=self.spatial_dimensions)
        dice = intersection / (union + self.epsilon)
        # If both inputs are empty the dice coefficient should be equal 1
        dice[union == 0] = 1
        return torch.mean(dice).item()


class Accuracy(SimpleMetric):
    def __init__(self, name: str = "accuracy"):
        """
        Accuracy metric for classification tasks.

        Args:
            name (str): The name of the metric.

        """
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Scalar:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        # Single value output, assume binary logits
        preds = (
            (logits > threshold).int() if len(logits.shape) == 1 or logits.shape[1] == 1 else torch.argmax(logits, 1)
        )
        target = target.cpu().detach()
        preds = preds.cpu().detach()
        return sklearn_metrics.accuracy_score(target, preds)


class BalancedAccuracy(SimpleMetric):
    def __init__(self, name: str = "balanced_accuracy"):
        """
        Balanced accuracy metric for classification tasks. Used for the evaluation of imbalanced datasets.

        For more information:

        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
        """
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        # assuming batch first
        assert logits.shape[0] == target.shape[0]
        target = target.cpu().detach()
        logits = logits.cpu().detach()
        y_true = target.reshape(-1)
        preds = np.argmax(logits, axis=1)
        return sklearn_metrics.balanced_accuracy_score(y_true, preds)


class RocAuc(SimpleMetric):
    def __init__(self, name: str = "ROC_AUC score"):
        """
        Area under the Receiver Operator Curve (AUCROC) metric for classification.

        For more information:

        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        """
        super().__init__(name)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> Scalar:
        assert logits.shape[0] == target.shape[0]
        prob = torch.nn.functional.softmax(logits, dim=1)
        prob = prob.cpu().detach()
        target = target.cpu().detach()
        y_true = target.reshape(-1)
        return sklearn_metrics.roc_auc_score(y_true, prob, average="weighted", multi_class="ovr")


class F1(SimpleMetric):
    def __init__(
        self,
        name: str = "F1 score",
        average: str | None = "weighted",
    ):
        """
        Computes the F1 score using the ``sklearn f1_score`` function. As such, the values of average correspond to
        those of that function.

        Args:
            name (str, optional): Name of the metric. Defaults to "F1 score".
            average (str | None, optional): Whether to perform averaging of the F1 scores and how. The values of this
                string corresponds to those of the ``sklearn f1_score function``. See:

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
        return sklearn_metrics.f1_score(y_true, preds, average=self.average)
