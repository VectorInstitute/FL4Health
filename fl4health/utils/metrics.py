import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum

import numpy as np
import torch
from flwr.common.typing import Metrics, Scalar
from sklearn import metrics as sklearn_metrics
from torchmetrics import Metric as TMetric

from fl4health.utils.typing import TorchPredType, TorchTargetType, TorchTransformFunction


class MetricPrefix(Enum):
    TEST_PREFIX = "test -"
    VAL_PREFIX = "val -"


TEST_NUM_EXAMPLES_KEY = f"{MetricPrefix.TEST_PREFIX.value} num_examples"
TEST_LOSS_KEY = f"{MetricPrefix.TEST_PREFIX.value} checkpoint"


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
        This method updates the state of the metric by appending the passed input and target pairing to their
        respective list.

        Args:
            input (torch.Tensor): The predictions of the model to be evaluated.
            target (torch.Tensor): The ground truth target to evaluate predictions against.

        Raises:
            NotImplementedError: To be defined in the classes extending this class.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, name: str | None) -> Metrics:
        """
        Compute metric on accumulated input and output over updates.

        Args:
            name (str | None): Optional name used in conjunction with class attribute name to define key in metrics
                dictionary.

        Raises:
            NotImplementedError: To be defined in the classes extending this class.

        Returns:
           Metrics: A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
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
        Thin wrapper on TorchMetric to make it compatible with our ``Metric`` interface.

        Args:
            name (str): The name of the metric.
            metric (TMetric): ``TorchMetric`` class based metric
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

    def compute(self, name: str | None) -> Metrics:
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
        pred_transforms: Sequence[TorchTransformFunction] | None = None,
        target_transforms: Sequence[TorchTransformFunction] | None = None,
        name: str | None = None,
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
            name (str | None, optional): Name of the Transformed Metric. If left as None defaults to metric.name
        """
        self.metric = metric
        self.pred_transforms = [] if pred_transforms is None else pred_transforms
        self.target_transforms = [] if target_transforms is None else target_transforms
        self.name = self.metric.name if name is None else name

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        for transform in self.pred_transforms:
            pred = transform(pred)

        for transform in self.target_transforms:
            target = transform(target)

        self.metric.update(pred, target)

    def compute(self, name: str | None) -> Metrics:
        # Change the name of the original metric class temporarily s
        original_metric_name = self.metric.name
        self.metric.name = self.name
        metrics_dict = self.metric.compute(name)
        self.metric.name = original_metric_name
        return metrics_dict

    def clear(self) -> None:
        return self.metric.clear()


class EMAMetric(Metric):
    def __init__(self, metric: Metric, name: str | None = None, smoothing_factor: float = 0.1):
        """
        Exponential Moving Average (EMA) metric wrapper to apply EMA to the computed metric.

        Args:
            metric (Metric): A FL4Health compatible metric
            name (str | None, optional): Name of the EMAMetric. If left as None will default to 'EMA_{metric.name}'.
            smoothing_factor (float, optional): Smoothing factor in range [0, 1] for the EMA. Smaller values increase
                smoothing by weighting previous scores more heavily. Defaults to 0.1.
        """
        self.metric = metric
        self.smoothing_factor = smoothing_factor
        self.previous_score: Metrics | None = None
        self.name = f"EMA_{self.metric.name}" if name is None else name

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        return self.metric.update(input, target)

    def compute(self, name: str | None = None) -> Metrics:
        # Compute current metric score.
        # Temporarily change name of the underlying metric so that we get the EMAMetric name in keys of metrics_dict
        metric_name = self.metric.name
        self.metric.name = self.name
        metrics_dict = self.metric.compute(name)
        self.metric.name = metric_name

        # Check if this is the first score
        if self.previous_score is None:
            self.previous_score = metrics_dict
            return metrics_dict

        # Otherwise compute EMA score for each 'metric' in Metrics dict
        for key, current_score in metrics_dict.items():
            previous_score = self.previous_score[key]
            if (
                not isinstance(current_score, str)
                and not isinstance(current_score, bytes)
                and not isinstance(previous_score, str)
                and not isinstance(previous_score, bytes)
            ):
                self.previous_score[key] = (
                    self.smoothing_factor * current_score + (1 - self.smoothing_factor) * previous_score
                )

        return self.previous_score

    def clear(self) -> None:
        # Clear accumulated inputs and targets but not the previous score
        return self.metric.clear()


# class HardDICE(Metric):
#     def __init__(
#         self,
#         name: str = "Hard-DICE",
#         along_axes: Sequence[int] = (0,),
#         ignore_background_axis: int | None = None,
#         ignore_null: bool = True,
#     ) -> None:
#         """
#         Computes the Mean DICE Coefficient between categorical (Hard) class predictions and targets.

#         Preds and targets passed to __call__ are assumed to contain only binary integers. Consider using
#         fl4health.utils.metrics.TransformsMetric to apply transforms to preds and targets before computing the DICE
#         score. For example you may need to use an argmax and one-hot-encoding function to convert predicted logits to
#         'hard' class predictions.

#         Args:

#             name (str): Name of the metric. Defaults to 'DICE'
#             along_axes (Sequence[int]): Sequence of indices specifying along which axes the individual DICE
#                 coefficients should be computed. The final DICE Score is the mean of these DICE coefficients. Defaults
#                 to (0,) since this is usually the batch dimension. If provided an empty tuple then a single DICE coefficient will be computed over all axes.
#             ignore_background_axis (int | None): If specified, the first channel of the specified axis is removed
#                 prior to computing the DICE coefficients. Useful for removing background channels. Defaults to None.
#             ignore_null (bool): If True, null dice coefficients are removed before returning the mean dice score. If
#                 False then null dice scores are set to 1. Null DICE scores are usually a result of the prediction and
#                 target both being all-zero (True Negatives). How this argument affects the final DICE score will vary
#                 depending along which axes the DICE coefficients were computed. Defaults to True.
#         """
#         self.ignore_background_axis = ignore_background_axis
#         self.along_axes = tuple([a for a in along_axes])
#         self.ignore_background_axis = ignore_background_axis
#         self.ignore_null = ignore_null
#         self.name = name

#         # We will accumulate boolean tensors for TP, FP and FN to reduce memory overhead
#         self.tp = torch.tensor([], dtype=torch.bool)
#         self.fp = torch.tensor([], dtype=torch.bool)
#         self.fn = torch.tensor([], dtype=torch.bool)

#     def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
#         # Assertions to prevent this metric being used improperly
#         assert preds.shape == targets.shape, (
#             f"Preds and targets must have the same shape but got {preds.shape} and {targets.shape} respectively."
#         )

#         # Remove the background channel from the axis specified by ignore_background_axis
#         if self.ignore_background_axis is not None:
#             indices = torch.arange(1, preds.shape[self.ignore_background_axis], device=preds.device)
#             preds = torch.index_select(preds, self.ignore_background_axis, indices)
#             targets = torch.index_select(targets, self.ignore_background_axis, indices)

#         # Save tp, fp and fn as boolean arrays to prevent memory build up
#         preds, targets = preds.bool(), targets.bool()  # Convert to boolean tensors
#         self.tp = torch.cat([self.tp.to(preds.device), preds * targets])
#         self.fp = torch.cat([self.fp.to(preds.device), preds * ~targets])
#         self.fn = torch.cat([self.fn.to(preds.device), ~preds * targets])

#     def _compute_dice_coefficients(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
#         # Compute union and intersection along specified axes
#         sum_axes = tuple([i for i in range(tp.ndim) if i not in self.along_axes])
#         numerator = 2 * tp.sum(dim=sum_axes)  # Equivalent to 2 times the intersection
#         denominator = (2 * tp + fp + fn).sum(dim=sum_axes)  # Equivalent to the union

#         # Prevent div by 0 by handling null scores
#         if self.ignore_null:  # Ignore dice scores that will be null by removing them from the union and intersection
#             numerator = numerator[denominator != 0]
#             denominator = denominator[denominator != 0]
#         else:  # Set dice scores that would be null to 1. This might be good if they are considered True Negatives.
#             numerator[denominator == 0] = 1
#             denominator[denominator == 0] = 1

#         # Compute dice coefficients and return
#         return numerator / denominator

#     def compute(self, name: str | None = None) -> Metrics:
#         # Compute dice coefficients and return mean DICE score
#         dice = self._compute_dice_coefficients(self.tp, self.fp, self.fn)
#         key = self.name if name is None else f"{name} - {self.name}"
#         return {key: torch.mean(dice).item()}

#     def clear(self) -> None:
#         # Reset accumulated tp, fp and fn's.
#         self.tp = torch.tensor([], dtype=torch.bool)
#         self.fp = torch.tensor([], dtype=torch.bool)
#         self.fn = torch.tensor([], dtype=torch.bool)


class SoftDICE(Metric):
    def __init__(
        self,
        name: str = "Soft-DICE",
        along_axes: Sequence[int] = (0,),
        ignore_background_axis: int | None = None,
        ignore_null: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Computes the Mean DICE Coefficient between class predictions and targets.

        Preds and targets passed to __call__ are assumed to have the same shape and contain elements in range [0, 1].
        For multiclass problems ensure they are both one-hot-encoded.

        Args:

            name (str): Name of the metric. Defaults to 'DICE'
            along_axes (Sequence[int]): Sequence of indices specifying along which axes the individual DICE
                coefficients should be computed. The final DICE Score is the mean of these DICE coefficients. Defaults
                to (0,) which is assumed to be the batch/sample dimension. If provided an empty tuple then a single
                DICE coefficient will be computed over all axes. Note that intermediate values must be stored in memory
                for each element along the specified axes, this may lead to memory build up in some instances.
            ignore_background_axis (int | None): If specified, the first channel of the specified axis is removed
                prior to computing the DICE coefficients. Useful for removing background channels. Defaults to None.
            ignore_null (bool): If True, null dice coefficients are removed before returning the mean dice score. If
                False then null dice scores are set to 1. Null DICE scores are usually a result of the prediction and
                target both being all-zero (True Negatives). How this argument affects the final DICE score will vary
                depending along which axes the DICE coefficients were computed. Defaults to True.
            dtype (torch.dtype, optional): The torch dtype to use when storing the intermediate true postive (tp),
                false positive (fp) and false negative (fn) sums. Must be a float if predictions are continious.
        """
        self.ignore_background_axis = ignore_background_axis
        self.along_axes = tuple([a for a in along_axes])
        self.ignore_background_axis = ignore_background_axis
        self.ignore_null = ignore_null
        self.name = name
        self.dtype = dtype
        self.tp_fp_fn_initialized = False

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Assertions to prevent this metric being used improperly
        assert preds.shape == targets.shape, (
            f"Preds and targets must have the same shape but got {preds.shape} and {targets.shape} respectively."
        )  # NOTE: It would be cool to add a utility function that attempts to infer the channel dimension if the
        # shapes are not the same and one-hot encodes the tensor with fewer dimensions.
        assert torch.min(preds) >= 0 and torch.max(preds) <= 1, "Expected preds to be in range [0, 1]."
        assert torch.min(targets) >= 0 and torch.max(targets) <= 1, "Expected targets to be in range [0, 1]."

        # Remove the background channel from the axis specified by ignore_background_axis
        if self.ignore_background_axis is not None:
            indices = torch.arange(1, preds.shape[self.ignore_background_axis], device=preds.device)
            preds = torch.index_select(preds, self.ignore_background_axis, indices)
            targets = torch.index_select(targets, self.ignore_background_axis, indices)

        # On the off chance were given booleans convert them to integers
        preds = preds.to(torch.uint8) if preds.dtype == torch.bool else preds
        targets = targets.to(torch.uint8) if targets.dtype == torch.bool else targets

        # Compute tp, fp and fn
        sum_axes = tuple([i for i in range(preds.ndim) if i not in self.along_axes])
        tp = (preds * targets).sum(dim=sum_axes, dtype=self.dtype)
        fp = (preds * (1 - targets)).sum(dim=sum_axes, dtype=self.dtype)
        fn = ((1 - preds) * targets).sum(dim=sum_axes, dtype=self.dtype)

        # If tp, fp and fn don't exist yet, then initialize them and exit function
        if not self.tp_fp_fn_initialized:
            self.tp, self.fp, self.fn = tp, fp, fn
            self.tp_fp_fn_initialized = True
            return

        # If the batch/sample dimension is in self.along_axes, we must concatenate the values; otherwise, we sum them
        self.tp = torch.cat([self.tp, tp], dim=0) if 0 in self.along_axes else self.tp + tp
        self.fp = torch.cat([self.fp, fp], dim=0) if 0 in self.along_axes else self.fp + fp
        self.fn = torch.cat([self.fn, fn], dim=0) if 0 in self.along_axes else self.fn + fn

    def _compute_dice_coefficients(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Compute union and intersection
        print(tp, fp, fn)
        numerator = 2 * tp  # Equivalent to 2 times the intersection
        denominator = 2 * tp + fp + fn  # Equivalent to the union

        # Prevent div by 0 by handling null scores
        if self.ignore_null:  # Ignore dice scores that will be null by removing them from the union and intersection
            numerator = numerator[denominator != 0]
            denominator = denominator[denominator != 0]
        else:  # Set dice scores that would be null to 1. This might be good if they are considered True Negatives.
            numerator[denominator == 0] = 1
            denominator[denominator == 0] = 1

        # Compute dice coefficients and return
        return numerator / denominator

    def compute(self, name: str | None = None) -> Metrics:
        # Compute dice coefficients and return mean DICE score
        dice = self._compute_dice_coefficients(self.tp, self.fp, self.fn)
        print(dice)
        key = self.name if name is None else f"{name} - {self.name}"
        return {key: torch.mean(dice).item()}

    def clear(self) -> None:
        # Reset accumulated tp, fp and fn's.
        del self.tp, self.fp, self.fn
        self.tp_fp_fn_initialized = False


class HardDICE(SoftDICE):
    def __init__(
        self,
        name: str = "Hard-DICE",
        along_axes: Sequence[int] = (0,),
        ignore_background_axis: int | None = None,
        ignore_null: bool = True,
        binarize: float | int | None = None,
    ) -> None:
        """
        Computes the Mean DICE Coefficient between categorical (Hard) class predictions and targets.

        Preds and targets passed to __call__ are assumed to have the same shape and contain elements in range [0, 1]. For multiclass problems ensure they are both one-hot-encoded.

        Args:

            name (str): Name of the metric. Defaults to 'DICE'
            along_axes (Sequence[int]): Sequence of indices specifying along which axes the individual DICE
                coefficients should be computed. The final DICE Score is the mean of these DICE coefficients. Defaults
                to (0,) since this is usually the batch dimension. If provided an empty tuple then a single DICE coefficient will be computed over all axes.
            ignore_background_axis (int | None): If specified, the first channel of the specified axis is removed
                prior to computing the DICE coefficients. Useful for removing background channels. Defaults to None.
            ignore_null (bool): If True, null dice coefficients are removed before returning the mean dice score. If
                False then null dice scores are set to 1. Null DICE scores are usually a result of the prediction and
                target both being all-zero (True Negatives). How this argument affects the final DICE score will vary
                depending along which axes the DICE coefficients were computed. Defaults to True.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the
                threshold are mapped to 0 and above are mapped to 1. If an integer is given, predictions are binarized
                based on the class with the highest prediction. Default of None leaves preds and targets unchanged.
        """
        # Use int64 to prevent overflow
        self.binarize = binarize
        super().__init__(name, along_axes, ignore_background_axis, ignore_null, dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if isinstance(self.binarize, float):
            preds = (preds > self.binarize).to(torch.uint8)
        elif isinstance(self.binarize, int):
            # Use argmax to get predicted class labels (hard_preds) and the one-hot-encode them.
            hard_preds = preds.argmax(self.binarize, keepdim=True)
            preds = torch.zeros_like(preds)
            preds.scatter_(self.binarize, hard_preds, 1)

        super().update(preds, targets)


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
        Balanced accuracy metric for classification tasks. Used for the evaluation of imbalanced datasets. For more
        information:

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
        self.metrics_per_prediction_type: dict[str, Sequence[Metric]] = {}

    def update(self, preds: TorchPredType, target: TorchTargetType) -> None:
        """
        Updates (or creates then updates) a list of metrics for each prediction type.

        Args:
            preds (TorchPredType): A dictionary of preds from the model
            target (TorchTargetType): The ground truth labels for the data. If target is a dictionary with more than
                one item, then each value in the preds dictionary is evaluated with the value that has the same key in
                the target dictionary. If target has only one item or is a ``torch.Tensor``, then the same target is
                used for all predictions.
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
            Metrics: dictionary containing computed metrics along with string identifiers for each prediction type.
        """
        all_results = {}
        for metrics_key, metrics in self.metrics_per_prediction_type.items():
            for metric in metrics:
                result = metric.compute(f"{self.metric_manager_name} - {metrics_key}")
                all_results.update(result)

        return all_results

    def clear(self) -> None:
        """
        Clears data accumulated in each metric for each of the prediction type.
        """
        for metrics_for_prediction_type in self.metrics_per_prediction_type.values():
            for metric in metrics_for_prediction_type:
                metric.clear()

    def reset(self) -> None:
        """
        Resets the metrics to their initial state.
        """
        # On next update, metrics will be recopied from self.original_metrics which are still in their initial state
        self.metrics_per_prediction_type = {}

    def check_target_prediction_keys_equal(
        self, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> None:
        assert target.keys() == preds.keys(), (
            "Received a dict with multiple targets, but the keys of the "
            "targets do not match the keys of the predictions. Please pass a "
            "single target or ensure the keys between preds and target are the same"
        )
