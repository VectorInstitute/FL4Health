import copy
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from flwr.common.typing import Metrics, Optional, Scalar
from sklearn import metrics as sklearn_metrics
from torchmetrics import Metric as TMetric

from fl4health.utils.typing import TorchPredType, TorchTargetType, TorchTransformFunction


class TestMetricPrefix(Enum):
    __test__ = False
    TEST_PREFIX = "test -"


TEST_NUM_EXAMPLES_KEY = f"{TestMetricPrefix.TEST_PREFIX.value} num_examples"
TEST_LOSS_KEY = f"{TestMetricPrefix.TEST_PREFIX.value} loss"


class Metric(ABC):
    def __init__(self, name: str) -> None:
        """
        Base abstract Metric class to extend for metric accumulation and computation.

        Args:
            name (str): Name of the metric.
        """
        self.name = name

    @abstractmethod
    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        This method updates the state of the metric by appending the passed input and target
        pairing to their respective list.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.

        Raises:
            NotImplementedError: To be defined in the classes extending this class.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, name: Optional[str]) -> Metrics:
        """
        Compute metric on accumulated input and output over updates.

        Args:
            name (Optional[str]): Optional name used in conjunction with class attribute name
                to define key in metrics dictionary.

        Raises:
            NotImplementedError: To be defined in the classes extending this class.

        Returns:
           Metrics: A dictionary of string and Scalar representing the computed metric
                and its associated key.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Resets metric.

        Raises:
            NotImplementedError: To be defined in the classes expending this class.
        """
        raise NotImplementedError


class TorchMetric(Metric):
    def __init__(self, name: str, metric: TMetric) -> None:
        """
        Thin wrapper on TorchMetric to make it compatible with our Metric interface.

        Args:
            name (str): The name of the metric.
            metric (TMetric): TorchMetric class based metric
        """
        super().__init__(name)
        self.metric = metric

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the state of the underlying TorchMetric.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.
        """
        self.metric.update(input, target.long())

    def compute(self, name: Optional[str]) -> Metrics:
        """
        Compute value of underlying TorchMetric.

        Args:
            name (Optional[str]): Optional name used in conjunction with class attribute name
                to define key in metrics dictionary.

        Returns:
           Metrics: A dictionary of string and Scalar representing the computed metric
                and its associated key.
        """
        result_key = f"{name} - {self.name}" if name is not None else self.name
        result = self.metric.compute().item()
        return {result_key: result}

    def clear(self) -> None:
        self.metric.reset()


class SimpleMetric(Metric, ABC):
    def __init__(self, name: str) -> None:
        """
        Abstract metric class with base functionality to update, compute and clear metrics.
        User needs to define __call__ method which returns metric given inputs and target.

        Args:
            name (str): Name of the metric.
        """
        super().__init__(name)
        self.accumulated_inputs: List[torch.Tensor] = []
        self.accumulated_targets: List[torch.Tensor] = []

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        """
        This method updates the state of the metric by appending the passed input and target
        pairing to their respective list.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.
        """
        self.accumulated_inputs.append(input)
        self.accumulated_targets.append(target)

    def compute(self, name: Optional[str] = None) -> Metrics:
        """
        Compute metric on accumulated input and output over updates.

        Args:
            name (Optional[str]): Optional name used in conjunction with class attribute name
                to define key in metrics dictionary.

        Raises:
            AssertionError: Input and target lists must be non empty.

        Returns:
            Metrics: A dictionary of string and Scalar representing the computed metric
                and its associated key.
        """

        assert len(self.accumulated_inputs) > 0 and len(self.accumulated_targets) > 0
        stacked_inputs = torch.cat(self.accumulated_inputs)
        stacked_targets = torch.cat(self.accumulated_targets)
        result = self.__call__(stacked_inputs, stacked_targets)
        result_key = f"{name} - {self.name}" if name is not None else self.name

        return {result_key: result}

    def clear(self) -> None:
        """
        Resets metrics by clearing input and target lists.
        """
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


class TransformsMetric(Metric):
    def __init__(
        self,
        metric: Metric,
        pred_transforms: Optional[Sequence[TorchTransformFunction]] = None,
        target_transforms: Optional[Sequence[TorchTransformFunction]] = None,
    ) -> None:
        """
        A thin wrapper class to allow transforms to be applied to preds and
        targets prior to calculating metrics. Transforms are applied in the order given

        Args:
            metric (Metric): A FL4Health compatible metric
            pred_transforms (Optional[Sequence[TorchTransformFunction]], optional): A
                list of transform functions to apply to the model predictions before
                computing the metrics. Each callable must accept and return a torch.
                Tensor. Use partial to set other arguments.
            target_transforms (Optional[Sequence[TorchTransformFunction]], optional): A
                list of transform functions to apply to the targets before computing
                the metrics. Each callable must accept and return a torch.Tensor. Use
                partial to set other arguments.
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

    def compute(self, name: Optional[str]) -> Metrics:
        return self.metric.compute(name)

    def clear(self) -> None:
        return self.metric.clear()


class BinarySoftDiceCoefficient(SimpleMetric):
    def __init__(
        self,
        name: str = "BinarySoftDiceCoefficient",
        epsilon: float = 1.0e-7,
        spatial_dimensions: Tuple[int, ...] = (2, 3, 4),
        logits_threshold: Optional[float] = 0.5,
    ):
        """
        Binary DICE Coefficient Metric with configurable spatial dimensions and logits threshold.

        Args:
            name (str): Name of the metric.
            epsilon (float): Small float to add to denominator of DICE calculation to avoid divide by 0.
            spatial_dimensions (Tuple[int, ...]): The spatial dimensions of the image within the prediction tensors.
                The default assumes that the images are 3D and have shape:
                batch_size, channel, spatial, spatial, spatial.
            logits_threshold: This is a threshold value where values above are classified as 1
                and those below are mapped to 0. If the threshold is None, then no thresholding is performed
                and a continuous or "soft" DICE coefficient is computed.
        """
        self.epsilon = epsilon
        self.spatial_dimensions = spatial_dimensions

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
        """
        Accuracy metric for classification tasks.

        Args:
            name (str): The name of the metric.

        """
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


class ROC_AUC(SimpleMetric):
    def __init__(self, name: str = "ROC_AUC score"):
        """
        Area under the Receiver Operator Curve (AUCROC) metric for classification. For more information:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
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
        return sklearn_metrics.f1_score(y_true, preds, average=self.average)


class MetricManager:
    def __init__(self, metrics: Sequence[Metric], metric_manager_name: str) -> None:
        """
        Class to manage a set of metrics associated to a given prediction type.

        Args:
            metrics (Sequence[Metric]): List of metric to evaluate predictions on.
            metric_manager_name (str): Name of the metric manager (ie train, val, test)
        """
        self.original_metrics = metrics
        self.metric_manager_name = metric_manager_name
        self.metrics_per_prediction_type: Dict[str, Sequence[Metric]] = {}

    def update(self, preds: TorchPredType, target: TorchTargetType) -> None:
        """
        Updates (or creates then updates) a list of metrics for each prediction type.

        Args:
            preds (TorchPredType): A dictionary of preds from the model
            target (TorchTargetType): The ground truth labels for the data. If
                target is a dictionary with more than one item, then each value
                in the preds dictionary is evaluated with the value that has
                the same key in the target dictionary. If target has only one
                item or is a torch.Tensor, then the same target is used for all
                predictions
        """
        if not self.metrics_per_prediction_type:
            self.metrics_per_prediction_type = {key: copy.deepcopy(self.original_metrics) for key in preds.keys()}

        # Check if there are multiple targets
        if isinstance(target, dict):
            if len(target.keys()) > 1:
                self.check_target_prediction_keys_equal(preds, target)
            else:  # There is only one target, get tensor from dict
                target = list(target.values())[0]
        for prediction_key, pred in preds.items():
            metrics_for_prediction_type = self.metrics_per_prediction_type[prediction_key]
            assert len(preds) == len(self.metrics_per_prediction_type)
            for metric_for_prediction_type in metrics_for_prediction_type:
                if isinstance(target, torch.Tensor):
                    metric_for_prediction_type.update(pred, target)
                else:
                    metric_for_prediction_type.update(pred, target[prediction_key])

    def compute(self) -> Metrics:
        """
        Computes set of metrics for each prediction type.

        Returns:
            Metrics: dictionary containing computed metrics along with string identifiers
                for each prediction type.
        """
        all_results = {}
        for metrics_key, metrics in self.metrics_per_prediction_type.items():
            for metric in metrics:
                result = metric.compute(f"{self.metric_manager_name} - {metrics_key}")
                all_results.update(result)

        return all_results

    def clear(self) -> None:
        """
        Clears metrics for each of the prediction type.
        """
        self.metrics_per_prediction_type = {}

    def check_target_prediction_keys_equal(
        self, preds: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> None:
        assert target.keys() == preds.keys(), (
            "Received a dict with multiple targets, but the keys of the "
            "targets do not match the keys of the predictions. Please pass a "
            "single target or ensure the keys between preds and target are the same"
        )
