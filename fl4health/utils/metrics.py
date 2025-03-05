import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from logging import WARNING

import numpy as np
import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics, Scalar
from sklearn import metrics as sklearn_metrics
from torchmetrics import Metric as TMetric

from fl4health.utils.typing import TorchPredType, TorchTargetType, TorchTransformFunction


class MetricPrefix(Enum):
    TEST_PREFIX = "test -"
    VAL_PREFIX = "val -"


TEST_NUM_EXAMPLES_KEY = f"{MetricPrefix.TEST_PREFIX.value} num_examples"
TEST_LOSS_KEY = f"{MetricPrefix.TEST_PREFIX.value} checkpoint"


def infer_channel_dim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    """Infers the channel dimension given two related tensors of different shapes.

    Generally useful for inferring the channel dimension when one tensor is one-hot-encoded and the other is not. The
    channel dimension is inferred by looking for dimensions that either are not the same size, or are not present int
    tensor 2. If a dimension adjacent to the

    Args:
        tensor1 (torch.Tensor): The reference tensor. Must have the same number of dimensions as tensor 2, or have
            exactly 1 more dimension (the channel/class dim).
        tensor2 (torch.Tensor): The non-reference tensor.

    Raises:
        AssertionError: If the the channel dimension cannot be inferred without ambigiuty.

    Returns:
        int: Index of the dimension along tensor 1 that is the channel/class dimensions
    """
    assert (
        tensor1.shape != tensor2.shape and (tensor1.ndim - tensor2.ndim) <= 1
    ), f"Could not infer the channel dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape})."

    # Infer channel dimension.
    idx2 = 0
    candidate_channels = []
    for idx1 in range(tensor1.ndim):
        if idx2 >= tensor2.ndim:  # Just in case its the last channel we need to avoid indexing error
            candidate_channels.append(idx1)
        elif tensor1.shape[idx1] == tensor2.shape[idx2]:
            idx2 += 1
        else:
            candidate_channels.append(idx1)
            if tensor1.ndim == tensor2.ndim:
                idx2 += 1

    assert len(candidate_channels) == 1, (
        f"Could not infer the channel dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
        "Found multiple axes that could be the channel dimension."
    )
    ch = candidate_channels[0]

    # Cover edge case where dim adjacent to channel dim has the same size.
    # We will mistakenly resolve only a single candidate channel when technically it is ambiguous.
    if tensor1.ndim > tensor2.ndim and ch > 0:
        assert tensor1.shape[ch] != tensor1.shape[ch - 1], (
            f"Could not infer the channel dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
            "A dimension adjacent to the channel dimension appears to have the same size."
        )

    # If tensors have same ndim but diff shape, then this only works if channel dim was empty for one of them
    if tensor1.ndim == tensor2.ndim:
        assert (tensor1.shape[ch] == 1) or (tensor2.shape[ch]) == 1, (
            f"Could not infer the channel dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
            "The inferred candidate dimension has different sizes on each tensor, was expecting one to be empty."
        )
    return ch


def align_pred_and_target_shapes(
    preds: torch.Tensor, targets: torch.Tensor, channel_dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attempts to correct shape mismatches between the given tensors by inferring which one to one-hot-encode.

    If both input tensors are not one-hot-encoded, then they are returned unchanged. If one is one-hot-encoded but not
    the other, then both are returned as one-hot-encoded tensors.

    Args:
        preds (torch.Tensor): The tensor with model predictions.
        targets (torch.Tensor): The tensor with model targets.
        channel_dim (int | None): Index of the channel dimension. If left as None then this method attempts to infer
            the channel dimension if it is needed.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The pred and target tensors respectively now ensured to have the same shape.
    """
    swapped = False
    if preds.shape == targets.shape:
        return preds, targets  # Shapes are already aligned.

    # If shapes are different then we assume one tensor is OHE and the other is not.
    # Tensor1 is the OHE reference and will not be modified, the other will be one-hot-encoded.
    if preds.ndim > targets.ndim:  # Preds must be one-hot encoded and targets are not
        tensor1, tensor2 = preds, targets
    else:
        # If targets have more dims than preds, then targets must be ohe and preds are not
        # If ndims are equal but shapes are not, then we can only determine reference tensor after finding channel dim.
        tensor1, tensor2 = targets, preds
        swapped = not swapped

    # Determine channel dimension. This method also has a bunch of necessary assertions.
    ch = infer_channel_dim(tensor1, tensor2) if channel_dim is None else channel_dim

    assert (
        tensor1.ndim - tensor2.ndim
    ) <= 1, f"Can not align pred and target tensors with shapes {preds.shape}, {targets.shape}"

    # Add channel dimension if there isn't one
    if tensor1.ndim != tensor2.ndim:
        tensor2 = tensor2.view((*tensor2.shape[:ch], 1, *tensor2.shape[ch:]))

    # Swap tensors on off chance that the first one had an empty dim and the second didn't
    if tensor1.shape[ch] < tensor2.shape[ch]:
        tensor1, tensor2 = tensor2, tensor1
        swapped = not swapped

    # One-hot-encode tensor2. We know at this point that it has an empty dim along channel axis
    if torch.is_floating_point(tensor2) and torch.frac(tensor2).sum() != 0:
        # If tensor is continious it must be binary classification and elements must be probabilities in range [0, 1].
        t2_ohe = torch.cat([1 - tensor2, tensor2], dim=ch)
    else:
        # If tensor2 is not continious then it must contain class labels. One hot encode the tensor.
        t2_ohe = torch.zeros(tensor1.shape, device=tensor1.device)
        t2_ohe.scatter_(ch, tensor2.to(torch.int64), 1)

    # Return modified tensors in their original positions.
    return (t2_ohe, tensor1) if swapped else (tensor1, t2_ohe)


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


class SimpleMetric(Metric, ABC):
    def __init__(self, name: str) -> None:
        """
        Abstract metric class with base functionality to update, compute and clear metrics. User needs to define
        ``__call__`` method which returns metric given inputs and target.

        WARNING: This class accumulates all the predictions and targets in memory throughout each FL server round. This
        may lead to out of memory (OOM) issues. Subclassing SimpleMetric is recommended only for quickly prototyping
        new metrics that have existing implementations in other packages. See the ROC_AUC class for an example.
        In other scenarious it is generally recommended to subclass the base Metric class instead and implement a
        custom update method that reduces the memory footprint of the metric. See the Accuracy clas for an example.

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


class ClassificationMetric(Metric):
    def __init__(
        self,
        name: str,
        along_axes: Sequence[int],
        dtype: torch.dtype,
        binarize: float | int | None = None,
        ignore_background: int | None = None,
        discard: Sequence[str] | None = None,
    ) -> None:
        """A Base class for classification metrics that can be computed using the true positives (tp), false positive
        (fp), false negative (fn') and true negative (tn) counts.

        On each update, the tp, fp, fn and tn counts are reduced along the specified axes before being accumulated into
        ``self.tp``, ``self.fp``, ``self.fn`` and ``self.tn`` respectively. This reduces the memory footprint required
        to compute metrics across rounds. The User needs to define the ``compute_from_counts`` method which returns a
        dictionary of Scalar metrics given the tp, fp, fn, and tn counts. The accumulated counts are reset by the
        ``clear`` method. If your subclass returns multiple metrics you may need to override the `__call__` method.

        NOTE: Preds and targets must have the same shape and only contain elements in range [0, 1]. If preds and
        targets passed to update method have different shapes, this class will attempt to infer the channel dimension
        and align the shapes by one-hot-encoding one of the tensors.

        Args:
            name (str): The name of the metric.
            along_axes (Sequence[int]): Sequence of indices specifying axes *along* which to accumulate tp, fp, fn and
                tn. The counts will be summed *over* the axes not specified. The 0th axis is assumed to be the batch or
                sample dimension. If provided an empty sequence, then the counts are scalar values computed *over* all
                axes.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continious, specify a
                float type. Otherwise specify an integer type to prevent overflow.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the
                threshold are mapped to 0 and above are mapped to 1. If an integer is given, predictions are binarized
                based on the class with the highest prediction. Default of None leaves preds unchanged.
            ignore_background (int | None): If specified, the first channel of the specified axis is removed prior to
                computing the counts. Useful for removing background channels. Defaults to None.
            discard (Sequence[str] | None, optional): One or several of ['tp', 'fp', 'fn', 'tn']. Specified counts
                will not be accumulated. Their associated attribute will remain as an empty pytorch tensor. Useful for
                reducing the memory footprint of metrics that do not use all of the counts in their computation
        """
        self.name = name
        self.along_axes = tuple([a for a in along_axes])
        self.dtype = dtype
        self.binarize = binarize
        self.ignore_background = ignore_background
        self.channel_dim = ignore_background  # If channel dim is None then it will be inferred

        # Parse discard argument
        count_ids = ["tp", "fp", "fn", "tn"]
        discard = [] if discard is None else discard
        for count_id in discard:
            assert count_id in count_ids, f"Found unexpected string in discard list. Expected one of {count_ids}"

        self.discard_tp = "tp" in discard
        self.discard_fp = "fp" in discard
        self.discard_fn = "fn" in discard
        self.discard_tn = "tn" in discard

        # Create intermediate tensors. Will be initialized with tensors of correct shape on first update.
        self.counts_initialized = False
        self.tp, self.fp, self.fn, self.tn = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    def binarize_tensor(self, input: torch.Tensor, binarize: float | int) -> torch.Tensor:
        """Converts continious 'soft' tensors into categorical 'hard' ones.

        Args:
            input (torch.Tensor): The tensor to binarize.
            binarize (float | int, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, elements below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, elements are binarized based on the class
                with the highest prediction. If binarize is None then the input is returned unchanged

        Returns:
            torch.Tensor: Either binarized tensor or if binarized was None then the input tensor.
        """
        if isinstance(binarize, float):
            return (input > binarize).to(torch.uint8)
        elif isinstance(binarize, int):  # NOTE: Technically this works even if preds are unnormalized.
            # Use argmax to get predicted class labels (hard_preds) and the one-hot-encode them.
            if binarize >= input.ndim:
                raise ValueError(
                    f"Cannot apply softmax to Tensor of shape {input.shape}."
                    " If preds are not one-hot-encoded set the binarize argument to a float threshold or None."
                )
            hard_input = input.argmax(binarize, keepdim=True)
            input = torch.zeros_like(input)
            input.scatter_(binarize, hard_input, 1)
            return input
        else:
            raise ValueError(f"Was expecting binarize argument to be either a float or an int. Got {type(binarize)}")

    def _transform_tensors(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does all the necessary transformations to preds and targets before computing the counts.

        This may or may not include binarizing the preds, one-hot-encoding either the preds or targets, removing the
        background channel, and or type conversions. Several assertions are also made to ensure inputs are as expected.
        """
        # Maybe convert continious 'soft' predictions into binary 'hard' predictions.
        preds = preds if self.binarize is None else self.binarize_tensor(preds, self.binarize)

        # Attempt automatically match pred and target shape.
        # Added mainly because previous implementations of metrics assumed preds to be OHE but targets not to be OHE
        # This supports any combination of hard/soft, OHE/not-OHE so long as channel dim can be inferred.
        preds, targets = align_pred_and_target_shapes(preds, targets, self.channel_dim)

        # Assertions to prevent this metric being used improperly.
        assert (
            preds.shape == targets.shape
        ), f"Preds and targets must have the same shape but got {preds.shape} and {targets.shape} respectively."
        assert torch.min(preds) >= 0 and torch.max(preds) <= 1, "Expected preds to be in range [0, 1]."
        assert torch.min(targets) >= 0 and torch.max(targets) <= 1, "Expected targets to be in range [0, 1]."

        # Remove the background channel from the axis specified by ignore_background_axis
        if self.ignore_background is not None:
            indices = torch.arange(1, preds.shape[self.ignore_background], device=preds.device)
            preds = torch.index_select(preds, self.ignore_background, indices)
            targets = torch.index_select(targets, self.ignore_background, indices)

        # On the off chance were given booleans convert them to integers
        preds = preds.to(torch.uint8) if preds.dtype == torch.bool else preds
        targets = targets.to(torch.uint8) if targets.dtype == torch.bool else targets

        return preds, targets

    def count_tp_fp_fn_tn(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given two tensors containing model predictions and targets, returns the number of true positives (tp), false
        positives (fp), false negatives (fn) and true negatives (tn).

        If any of the tp, fp, fn or tn counts were specified to be discarded during initialization of the class, then
        that count will not be computed and an empty tensor will be returned in its place. The counts are summed along
        the axes specified in self.along_axes and there shape will be [pred.shape[a] for a in self.along_axes].

        Args:
            preds (torch.Tensor): Tensor containing model predictions. Must be the same shape and format as targets.
            targets (torch.Tensor): Tensor containing prediction targets. Must be same shape as preds.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors containing the counts along the
                specified dimensions for each of tp, fp, fn, and tn respectively.
        """
        # Compute tp, fp and fn
        sum_axes = tuple([i for i in range(preds.ndim) if i not in self.along_axes])

        # Compute counts. If were ignoring a count, set it as an empty tensor to avoid downstream errors.
        tp = (preds * targets) if not self.discard_tp else torch.tensor([])
        fp = (preds * (1 - targets)) if not self.discard_fp else torch.tensor([])
        fn = ((1 - preds) * targets) if not self.discard_fn else torch.tensor([])
        tn = ((1 - preds) * (1 - targets)) if not self.discard_tn else torch.tensor([])

        # Sum along specified axes only if there are axes to sum over.
        tp = tp.sum(sum_axes, dtype=self.dtype) if len(sum_axes) > 0 and not self.discard_tp else tp
        fp = fp.sum(sum_axes, dtype=self.dtype) if len(sum_axes) > 0 and not self.discard_fp else fp
        fn = fn.sum(sum_axes, dtype=self.dtype) if len(sum_axes) > 0 and not self.discard_fn else fn
        tn = tn.sum(sum_axes, dtype=self.dtype) if len(sum_axes) > 0 and not self.discard_tn else tn

        return tp, fp, fn, tn

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates the existing tp, fp, fn and tn counts with new counts computed from preds and targets."""
        # Transform preds and targets as necessary/specified before computing counts
        preds, targets = self._transform_tensors(preds, targets)

        # Get tp, fp, fn and tn counts for current update. If a count is discarded, then an empty tensor is returned.
        tp, fp, fn, tn = self.count_tp_fp_fn_tn(preds, targets)

        # If this is first update since init or clear, initialize counts attributes and exit function
        if not self.counts_initialized:
            self.tp, self.fp, self.fn, self.tn = tp, fp, fn, tn
            self.counts_initialized = True
            return

        # If the batch/sample dimension is in self.along_axes, we must concatenate the values; otherwise, we sum them
        self.tp = torch.cat([self.tp, tp], dim=0) if 0 in self.along_axes else self.tp + tp
        self.fp = torch.cat([self.fp, fp], dim=0) if 0 in self.along_axes else self.fp + fp
        self.fn = torch.cat([self.fn, fn], dim=0) if 0 in self.along_axes else self.fn + fn
        self.tn = torch.cat([self.tn, tn], dim=0) if 0 in self.along_axes else self.tn + tn

    def clear(self) -> None:
        # Reset accumulated tp, fp and fn's. They will be initialized with correct shape on next update
        self.tp, self.fp, self.fn, self.tn = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        self.counts_initialized = False

    def compute(self, name: str | None = None) -> Metrics:
        """Computes the metrics from the currently saved counts"""
        metrics = self.compute_from_counts(tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn)
        if name is not None:
            metrics = {f"{name} - {k}": v for k, v in metrics.items()}
        return metrics

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> Scalar:
        """Convenience function for computing the metric on given preds and targets without modifying class state.

        If there are multiple scalar values in the Metrics dictionary returned by `self.compute_from_counts` then this
        method will try to return the mean. If it can not it returns the first value in the dictionary.
        """
        # Transform preds and targets as necessary/specified before computing counts
        preds, targets = self._transform_tensors(preds, targets)

        # Get tp, fp, fn and tn counts for current update
        tp, fp, fn, tn = self.count_tp_fp_fn_tn(preds, targets)

        metrics = self.compute_from_counts(tp, fp, fn, tn)

        if all([isinstance(v, (int, float)) for v in metrics.values()]):
            # If there are multiple metric scalars then try to return the mean for the call function
            return float(np.mean(list(metrics.values())))  # type: ignore
        else:  # Otherwise just return the first value
            if len(metrics) > 1:
                log(WARNING, "Could not aggregate all metrics from compute_from_counts. Defaulting to first in dict.")
            return list(metrics.values())[0]

    def compute_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> Metrics:
        raise NotImplementedError


class Accuracy(ClassificationMetric):
    def __init__(
        self,
        name: str = "Accuracy",
        along_axes: Sequence[int] = (0,),
        binarize: float | int | None = 1,
        exact_match: bool = True,
        ignore_background: int | None = None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        """
        Memory efficient accuracy metric.

        Computes accuracy from true positives (tp), false positive (fp), false negative (fn') and true negative (tn)
        counts so that preds and targets don't need to be accumulated in memory.

        NOTE: Preds and targets must have the same shape and only contain elements in range [0, 1]. If preds and
        targets passed to update method have different shapes, this class will attempt to infer the channel dimension
        and align the shapes by one-hot-encoding one of the tensors.

        Args:
            name (str, optional): The name of the metric.
            along_axes (Sequence[int], optional): Sequence of indices specifying the axes *along* which to compute the
                individual accuracy scores which will then be averaged to produce the final accuracy score. If given an
                empty tuple, will compute a single accuracy score *over* all dimensions. Note that the individual
                accuracy scores must be stored in memory until cleared, this may cause memory build up in some
                instances. Default is to compute along the first dimension which is assumed to be the batch/n_samples
                dimension.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction. If None leaves preds unchanged. Default is 1 since this is usually the
                channel dimension.
            exact_match (bool): If True computes the 'Subset Accuracy'/'Exact Match Ratio'. Individual accuracies that
                are not prefect/exact (== 1) are set to 0 before computing the final accuracy score. This is useful for
                multilabel tabular classification. Defaults to True.
            ignore_background (int | None): If not None, the first channel of the specified axis is removed prior to
                computing the counts. Useful for removing background channels. Defaults to None.
            dtype (torch.dtype): Dtype used to store tp, fp, fn and tn counts in memory on on `self.update`.


        NOTE: To make this behave like the previous accuracy implementation using sklearn and SimpleMetric, use args
        {'binarize': 1, 'along_axes': [0], 'exact_match': True}. Replace binarize with 0.5 if preds are binary *and*
        not one-hot-encoded.
        """
        self.exact_match = exact_match
        super().__init__(
            name=name, along_axes=along_axes, dtype=dtype, binarize=binarize, ignore_background=ignore_background
        )

    def compute_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> Metrics:
        # Compute individual accuracy scores. Div by 0 shouldn't be possible here.
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        # If exact match is true, set all scores below 1 to 0.
        accuracy = (accuracy == 1).to(torch.float) if self.exact_match else accuracy
        return {self.name: torch.mean(accuracy).item()}


class Recall(ClassificationMetric):
    def __init__(
        self,
        name: str = "Recall",
        along_axes: Sequence[int] = (),
        binarize: float | int | None = None,
        ignore_background: int | None = None,
        zero_division: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Memory efficient recall metric (aka sensitivity).

        NOTE: Preds and targets must have the same shape and only contain elements in range [0, 1]. If preds and
        targets passed to update method have different shapes, this class will attempt to infer the channel dimension
        and align the shapes by one-hot-encoding one of the tensors.

        Args:
            name (str, optional): The name of the metric.
            along_axes (Sequence[int], optional): Sequence of indices specifying axes *along* which to compute the
                Recall. The recall scores will be summed *over* the axes not specified. The final recall score will be
                the mean of these recalls. The 0th axis is assumed to be the batch/sample dimension. If provided an
                empty sequence, then a single recall score is computed *over* all axes.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the
                threshold are mapped to 0 and above are mapped to 1. If an integer is given, predictions are binarized
                based on the class with the highest prediction. Default of None leaves preds unchanged.
            ignore_background (int | None, optional): If specified, the first channel of the specified axis is removed
                prior to computing the counts. Useful for removing background channels. Defaults to None.
            zero_division (float | None, optional): Set what the individual recall score should be when there is a
                zero division (only negative cases are present). If None, the resultant recall scores will be excluded
                from the average/final recall score.
            dtype (torch.dtype, optional): Dtype used to store tp, fp, fn and tn counts in memory on on `self.update`.
        """
        self.zero_division = zero_division
        super().__init__(
            name=name,
            along_axes=along_axes,
            dtype=dtype,
            binarize=binarize,
            ignore_background=ignore_background,
            discard=["fp", "tn"],
        )

    def compute_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> Metrics:
        # Compute denominator and remove or replace scores that will be null.
        denominator = tp + fn
        if self.zero_division is None:
            tp = tp[denominator != 0]
            denominator = denominator[denominator != 0]
        else:
            tp[denominator == 0] = self.zero_division
            denominator[denominator == 0] = 1

        # Compute recall scores and return mean
        recall = tp / denominator
        return {self.name: torch.mean(recall).item()}


class Dice(ClassificationMetric):
    def __init__(
        self,
        name: str = "Dice",
        along_axes: Sequence[int] = (0,),
        binarize: float | int | None = None,
        ignore_background: int | None = None,
        zero_division: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Computes the Mean DICE Coefficient between class predictions and targets.

        NOTE: Preds and targets must have the same shape and only contain elements in range [0, 1]. If preds and
        targets passed to update method have different shapes, this class will attempt to infer the channel dimension
        and align the shapes by one-hot-encoding one of the tensors.

        Args:

            name (str): Name of the metric. Defaults to 'Soft-DICE'
            along_axes (Sequence[int], optional): Sequence of indices specifying along which axes the individual DICE
                coefficients should be computed. The final DICE Score is the mean of these DICE coefficients. Defaults
                to (0,) which is assumed to be the batch/sample dimension. If provided an empty tuple then a single
                DICE coefficient will be computed over all axes. Note that intermediate values must be stored in memory
                for each element along the specified axes, this may lead to memory build up in some instances.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction. Default of None leaves preds unchanged and computes a 'Soft' Dice score,
                otherwise metric is equivalent to a 'Hard' Dice score.
            ignore_background (int | None, optional): If specified, the first channel of the specified axis is removed
                prior to computing the DICE coefficients. Useful for removing background channels. Defaults to None.
            zero_division (float | None, optional): Set what the individual dice coefficients should be when there is
                a zero division (only true negatives present). How this argument affects the final DICE score will vary
                depending along which axes the DICE coefficients were computed. If left as None, the resultant dice
                coefficients will be excluded from the average/final dice score.
            dtype (torch.dtype, optional): Dtype used to store tp, fp, fn and tn counts in memory on on `self.update`.
        """
        super().__init__(
            name=name,
            along_axes=along_axes,
            dtype=dtype,
            binarize=binarize,
            ignore_background=ignore_background,
            discard=["tn"],
        )
        self.zero_division = zero_division

    def _compute_dice_coefficients(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        # Compute union and intersection
        numerator = 2 * tp  # Equivalent to 2 times the intersection
        denominator = 2 * tp + fp + fn  # Equivalent to the union

        # Remove or replace dice score that will be null due to zero division
        if self.zero_division is None:
            numerator = numerator[denominator != 0]
            denominator = denominator[denominator != 0]
        else:
            numerator[denominator == 0] = self.zero_division
            denominator[denominator == 0] = 1

        # Return individual dice coefficients
        return numerator / denominator

    def compute_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> Metrics:
        # compute dice coefficients and return mean
        dice = self._compute_dice_coefficients(tp, fp, fn)
        return {self.name: torch.mean(dice).item()}


class HardDice(Dice):
    def __init__(
        self,
        name: str = "HardDice",
        along_axes: Sequence[int] = (0,),
        ignore_background: int | None = None,
        zero_division: float | None = None,
        binarize: float | int | None = None,
    ) -> None:
        """
        Computes the Mean DICE Coefficient between categorical (Hard) class predictions and targets.

        NOTE: Preds and targets must have the same shape and only contain elements in range [0, 1]. If preds and
        targets passed to update method have different shapes, this class will attempt to infer the channel dimension
        and align the shapes by one-hot-encoding one of the tensors. The binarize argument can be used to
        convert incoming continious ('soft') predictions in to categorical ('hard') predictions.

        Args:

            name (str): Name of the metric. Defaults to 'DICE'
            along_axes (Sequence[int], optional): Sequence of indices specifying *along* which axes the individual DICE
                coefficients should be computed. The final DICE Score is the mean of these DICE coefficients computed
                *over* the dimensions not specified. Defaults to (0,) since this is usually the batch dimension. If
                provided an empty tuple then a single DICE coefficient will be computed *over* all axes.
            ignore_background (int | None, optional): If specified, the first channel of the specified axis is removed
                prior to computing the DICE coefficients. Useful for removing background channels. Defaults to None.
            zero_division (float | None, optional): Set what the individual dice coefficients should be when there is
                a zero division (only true negatives present). How this argument affects the final DICE score will vary
                depending along which axes the DICE coefficients were computed. If left as None, the resultant dice
                coefficients will be excluded from the average/final dice score.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the
                threshold are mapped to 0 and above are mapped to 1. If an integer is given, predictions are binarized
                based on the class with the highest prediction. Default of None leaves preds and targets unchanged.
        """
        # This subclass used to do more but now just sets dtype to int64 to prevent overflow.
        super().__init__(
            name=name,
            along_axes=along_axes,
            binarize=binarize,
            ignore_background=ignore_background,
            zero_division=zero_division,
            dtype=torch.int64,
        )


class BalancedAccuracy(Recall):
    def __init__(
        self,
        name: str = "balanced_accuracy",
        channel_dim: int = 1,
        binarize: float | bool = True,
        ignore_background: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Balanced accuracy metric for classification tasks. Used for the evaluation of imbalanced datasets.

        Balanced accuracy is defined as the average recall obtained on each class.

        Args:
            name (str): The name of the metric.
            channel_dim (int, optional): Index specifying the axis representing the channel dimension. Defaults to 1.
            binarize (float | bool, optional): If a float is given, predictions below the
                value are mapped to 0 and above are mapped to 1. If True, predictions are binarized
                based on the class with the highest prediction. Defaults to True.
            ignore_background (bool, optional): If True, the first channel of the channel axis is removed prior to
                computing the counts. Useful for removing background channels. Defaults to False.
            dtype (torch.dtype, optional): Dtype used to store tp, fp, fn and tn counts in memory on on `self.update`.

        NOTE: To get this to behave like the previous implementation of balanced accuracy using sklearn and
        SimpleMetric, use kwargs {'channel_dim': 1, 'binarize': True, 'ignore_background': False}.
        """
        # We simplified the binarize metric since channel dim is always known. Must reformat before passing to parent.
        if isinstance(binarize, float):
            binarize_arg = binarize
        elif binarize:
            binarize_arg = int(channel_dim)
        else:
            binarize_arg = None

        super().__init__(
            name=name,
            along_axes=[channel_dim],
            dtype=dtype,
            binarize=binarize_arg,
            ignore_background=channel_dim if ignore_background else None,
        )

        # We override the channe dim attribute since this subclass forces it to be known.
        # Can prevent rare instances where channel dim is unable to be automatically inferred.
        self.channel_dim = channel_dim


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


class F1(ClassificationMetric):
    def __init__(
        self,
        name: str = "F1",
        along_axes: Sequence[int] = (),
        binarize: float | int | None = None,
        weighted: bool = False,
        zero_divison: float | None = None,
        ignore_background: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Computes the F1 Score.

        Args:
            name (str, optional): The name of the metric.
            along_axes (Sequence[int], optional): Sequence of indices specifying axes *along* which to compute the F1
                score. The F1 scores will be computed *over* the axes not specified and then averaged to produce the
                final F1 score. The 0th axis is assumed to be the batch/sample dimension. If provided an empty
                sequence, then a single F1 score is computed *over* all axes.
            binarize (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the channel/class dimension. If a float is given, predictions below the
                threshold are mapped to 0 and above are mapped to 1. If an integer is given, predictions are binarized
                based on the class with the highest prediction. Default of None leaves preds unchanged.
            weighted (bool, optional): If True, weights each of the individual F1 scores by their support (the number
                of true positives and false negatives) before averaging them to compute the final F1 score.
            zero_division (float | None, optional): Set what the individual F1 scores should be when there is a zero
                division (only true negatives present). How this argument affects the final F1 score will vary
                depending along which axes the individual scores are computed. If left as None, the resultant F1 scores
                will be excluded from the average/final F1 score.
            ignore_background (int | None, optional): If specified, the first channel of the specified axis is removed
                prior to computing the counts. Useful for removing background channels. Defaults to None.
            dtype (torch.dtype, optional): The dtype to store the recall scores as in memory. Defaults

        NOTE: To get this metric to behave like the previous implementation using sklearn and SimpleMetric, use kwargs
        {'along_axes': 1, 'binarize': 1, 'ignore_background': None, 'weighted': True, 'zero_division': 0.0}. Setting
        weighted to False the metric behave like the 'macro' averaging.
        """
        self.weighted = weighted
        self.zero_division = zero_divison
        super().__init__(
            name=name,
            along_axes=along_axes,
            dtype=dtype,
            binarize=binarize,
            ignore_background=ignore_background,
            discard=["tn"],
        )

    def compute_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> Metrics:
        # Calculate numerator and denominator
        numerator = 2 * tp
        denominator = 2 * tp + fp + fn

        # Remove or replace null F1 scores
        if self.zero_division is None:
            numerator = numerator[denominator != 0]
            denominator = denominator[denominator != 0]
        else:
            numerator[denominator == 0] = self.zero_division
            denominator[denominator == 0] = 1

        # Calculate individual F1 scores and aggregate into final score.
        f1 = numerator / denominator
        if self.weighted:
            f1 = (f1 * (tp + fn) / (tp.sum() + fn.sum())).sum()
        else:
            f1 = torch.mean(f1)
        return {self.name: f1.item()}


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
