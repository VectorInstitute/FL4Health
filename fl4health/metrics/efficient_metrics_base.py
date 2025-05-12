from abc import ABC, abstractmethod
from enum import Enum
from logging import INFO, WARNING

import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics, Scalar

from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metrics_utils import threshold_tensor
from fl4health.metrics.utils import align_pred_and_target_shapes


class MetricOutcome(Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_NEGATIVE = "false_negative"


class ClassificationMetric(Metric, ABC):
    def __init__(
        self,
        name: str,
        label_dim: int | None,
        batch_dim: int | None,
        dtype: torch.dtype,
        threshold: float | int | None,
        discard: set[MetricOutcome] | None,
    ) -> None:
        """
        A Base class for efficiently computing classification metrics that can be calculated using the true positives
        (tp), false positive (fp), false negative (fn) and true negative (tn) counts.

        How these values are counted is left to the inheriting class along with how they are composed together for the
        final metric score. There are two classes inheriting from this class to form the basis of efficient
        classification metrics: BinaryClassificationMetric and MultiClassificationMetric. These handle implementation
        of the ``count_tp_fp_fn_tn`` method.

        On each update, the true_positives, false_positives, false_negatives and true_negatives counts for the
        provided predictions and targets are accumulated into ``self.true_positives``, ``self.false_positives``,
        ``self.false_negatives`` and ``self.true_negatives``, respectively. This reduces the memory footprint
        required to compute metrics across rounds. The user needs to define the ``compute_from_counts`` method which
        returns a dictionary of Scalar metrics given the ``true_positives``, ``false_positives``, ``false_negatives``,
        and  ``true_negatives`` counts. The accumulated counts are reset by the ``clear`` method. If your subclass
        returns multiple metrics you may need to also override the ``__call__`` method.

        If the predictions provided are continuous in value, then the associated counts will also be continuous
        ("soft"). For example, with a target of 1, a prediction of 0.8 contributes 0.8 to the true_positives count and
        0.2 to the false_negatives.

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        the argument of this class to be as such.

        Args:
            name (str): The name of the metric.
            label_dim (int | None, optional): Specifies which dimension in the provided tensors corresponds to the
                label dimension.
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample INDIVIDUALLY.

                NOTE: The resulting counts will always be presented batch dimension first, then label dimension,
                regardless of input shape.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow.
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Setting to None leaves preds unchanged.
            discard (set[MetricOutcome] | None, optional): One or several of MetricOutcome values. Specified outcome
                counts will not be accumulated. Their associated attribute will remain as an empty pytorch tensor.
                Useful for reducing the memory footprint of metrics that do not use all of the counts in their
                computation.
        """
        self.name = name
        self.dtype = dtype
        self.threshold = threshold
        self.label_dim = label_dim
        self.batch_dim = batch_dim

        if self.label_dim is not None:
            if isinstance(self.threshold, int) and self.threshold != self.label_dim:
                log(
                    WARNING,
                    f"Specified threshold dimension: {threshold} is not the same as the label_dim: "
                    f"{label_dim}. This is atypical and may produce undesired behavior",
                )
            if self.batch_dim is not None and self.label_dim == self.batch_dim:
                raise ValueError(f"The label and batch dimensions must differ but got {self.label_dim}")

        # Parse discard argument
        discard = set() if discard is None else discard

        self.discard_tp = MetricOutcome.TRUE_POSITIVE in discard
        self.discard_fp = MetricOutcome.FALSE_POSITIVE in discard
        self.discard_fn = MetricOutcome.FALSE_NEGATIVE in discard
        self.discard_tn = MetricOutcome.TRUE_NEGATIVE in discard

        # Create intermediate tensors. Will be initialized with tensors of correct shape on first update.
        self.counts_initialized = False
        self.true_positives, self.false_positives, self.false_negatives, self.true_negatives = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

    def _transform_tensors(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given the predictions and targets this function performs two possible transformations. The first is to map
        boolean tensors to integers for computation. The second is to potentially threshold the predictions if
        self.threshold is not None

        NOTE: This is a common implementation meant to be called (or overridden) by inheriting classes.

        Args:
            preds (torch.Tensor): Predictions tensor
            targets (torch.Tensor): Targets tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Potentially transformed predictions and targets tensors, in that order
        """

        # On the off chance were given booleans convert them to integers
        preds = preds.to(torch.uint8) if preds.dtype == torch.bool else preds
        targets = targets.to(torch.uint8) if targets.dtype == torch.bool else targets

        # Maybe threshold predictions into 'hard' predictions.
        preds = preds if self.threshold is None else threshold_tensor(preds, self.threshold)
        return preds, targets

    def _assert_correct_ranges(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Ensures that the prediction and target tensors are within the expected ranges for computation.

        NOTE: This is a common implementation meant to be called (or overridden) by inheriting classes.

        Args:
            preds (torch.Tensor): Predictions tensor
            targets (torch.Tensor): Targets tensor
        """
        assert torch.min(preds) >= 0 and torch.max(preds) <= 1, "Expected preds to be in range [0, 1]."
        assert torch.min(targets) >= 0 and torch.max(targets) <= 1, "Expected targets to be in range [0, 1]."

    def _prepare_counts_from_preds_and_targets(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute counts. If were ignoring a count, set it as an empty tensor to avoid downstream errors.

        NOTE: preds and targets are assumed to be in range [0, 1]. Otherwise the computations below may produce
        unexpected results.

        Args:
            preds (torch.Tensor): Predictions tensor
            targets (torch.Tensor): Targets tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: True positive, false positive,
                false negative, and true negative indications for each of the predictions in the provided tensors.
        """

        true_positives = (preds * targets) if not self.discard_tp else torch.tensor([])
        false_positives = (preds * (1 - targets)) if not self.discard_fp else torch.tensor([])
        false_negatives = ((1 - preds) * targets) if not self.discard_fn else torch.tensor([])
        true_negatives = ((1 - preds) * (1 - targets)) if not self.discard_tn else torch.tensor([])
        return true_positives, false_positives, true_negatives, false_negatives

    def _sum_along_axes_or_discard(
        self,
        sum_axes: tuple[int, ...],
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Provided a set of axes over which to reduce by summation. These sums are applied to each of the
        provided tensors.

        Args:
            sum_axes (tuple[int, ...]): The dimension or dimensions to reduce. If empty, all dimensions are reduced.
            true_positives (torch.Tensor): Tensor with entry of 1 indicating a true positive for a pred/target pair
            false_positives (torch.Tensor): Tensor with entry of 1 indicating a false positive for a pred/target pair
            true_negatives (torch.Tensor): Tensor with entry of 1 indicating a true negative for a pred/target pair
            false_negatives (torch.Tensor): Tensor with entry of 1 indicating a false negative for a pred/target pair

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors reduced over the specified axes
                in the order ``true_positives``, ``false_positives``, ``true_negatives``, ``false_negatives``
        """
        true_positives = true_positives.sum(sum_axes, dtype=self.dtype) if not self.discard_tp else true_positives
        false_positives = false_positives.sum(sum_axes, dtype=self.dtype) if not self.discard_fp else false_positives
        false_negatives = false_negatives.sum(sum_axes, dtype=self.dtype) if not self.discard_fn else false_negatives
        true_negatives = true_negatives.sum(sum_axes, dtype=self.dtype) if not self.discard_tn else true_negatives
        return true_positives, false_positives, true_negatives, false_negatives

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the existing self.true_positive, self.false_positive, self.false_negative and self.true_negative
        counts with new counts computed from preds and targets.

        NOTE: In the implementation, this function implicitly assumes that the child classes inheriting from this
        class have implemented `count_tp_fp_fn_tn` such that, if `self.batch_dim` is not None, the counts are returned
        with shapes such that the batch dimension comes FIRST for the counts.

        Args:
            preds (torch.Tensor): Predictions tensor
            targets (torch.Tensor): Targets tensor
        """
        # Get tp, fp, tn and fn counts for current update. If a count is discarded, then an empty tensor is returned.
        tp, fp, tn, fn = self.count_tp_fp_tn_fn(preds, targets)

        # If this is first update since init or clear, initialize counts attributes and exit function
        if not self.counts_initialized:
            self.true_positives, self.false_positives, self.true_negatives, self.false_negatives = tp, fp, tn, fn
            self.counts_initialized = True
            return

        # If batch_dim has been specified, we accumulate counts for EACH INSTANCE seen throughout the updates, such
        # that each of the counts is a 2D tensor of row length equal to the samples seen.
        # NOTE: This ASSUMES the batch dimension comes FIRST for the counts.
        # Otherwise, the counts are 1D tensors with length equal to the number of classes (or possibly 1 if using the
        # BinaryClassificationMetric)
        self.true_positives = (
            torch.cat([self.true_positives, tp], dim=0) if self.batch_dim is not None else self.true_positives + tp
        )
        self.false_positives = (
            torch.cat([self.false_positives, fp], dim=0) if self.batch_dim is not None else self.false_positives + fp
        )
        self.true_negatives = (
            torch.cat([self.true_negatives, tn], dim=0) if self.batch_dim is not None else self.true_negatives + tn
        )
        self.false_negatives = (
            torch.cat([self.false_negatives, fn], dim=0) if self.batch_dim is not None else self.false_negatives + fn
        )

    def clear(self) -> None:
        """
        Reset accumulated tp, fp, fn and tn's. They will be initialized with correct shape on next update
        """
        self.true_positives = torch.tensor([])
        self.false_positives = torch.tensor([])
        self.false_negatives = torch.tensor([])
        self.true_negatives = torch.tensor([])
        self.counts_initialized = False

    def compute(self, name: str | None = None) -> Metrics:
        """
        Computes the metrics from the currently saved counts using the compute_from_counts function defined in
        inheriting classes

        Args:
            name (str | None): Optional name used in conjunction with class attribute name to define key in metrics
                dictionary.

        Returns:
            Metrics: A dictionary of string and ``Scalar`` representing the computed metric and its associated key.
        """
        metrics = self.compute_from_counts(
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=self.true_negatives,
            false_negatives=self.false_negatives,
        )
        if name is not None:
            metrics = {f"{name} - {k}": v for k, v in metrics.items()}
        return metrics

    @abstractmethod
    def count_tp_fp_tn_fn(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given two tensors containing model predictions and targets, returns the number of true positives (tp), false
        positives (fp), true negatives (tn), and false negatives (fn).

        The shape of these counts depends on the values of ``self.batch_dim`` and ``self.label_dim`` are specified
        and the implementation of the inheriting class.

        NOTE: Inheriting classes must implement additional functionality on top of this class. For example, any
        preprocessing that needs to be done to preds and targets should be done in the inheriting function. Any post
        processing should also be done there. See implementations in the `BinaryClassificationMetric` or
        `MultiClassificationMetric` class for examples.

        If any of the true positives, false positives, true negative, or false negative counts were specified to be
        discarded during initialization of the class, then that count will not be computed and an empty tensor will be
        returned in its place.

        Args:
            preds (torch.Tensor): Tensor containing model predictions. Must be the same shape as targets
            targets (torch.Tensor): Tensor containing prediction targets. Must be same shape as preds.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors containing the counts along the
                specified dimensions for each of true positives, false positives, false negative, and true negative
                respectively.
        """
        # Compute counts. If we're ignoring a count, set it as an empty tensor to avoid downstream errors.
        true_positives, false_positives, true_negatives, false_negatives = self._prepare_counts_from_preds_and_targets(
            preds, targets
        )

        axes_to_ignore: set[int] = {self.label_dim} if self.label_dim is not None else set()
        if self.batch_dim is not None:
            axes_to_ignore.add(self.batch_dim)
        sum_axes = tuple([i for i in range(preds.ndim) if i not in axes_to_ignore])

        # If sum_axes is empty, then we have nothing to do.
        if len(sum_axes) != 0:
            true_positives, false_positives, true_negatives, false_negatives = self._sum_along_axes_or_discard(
                sum_axes, true_positives, false_positives, true_negatives, false_negatives
            )

        return true_positives, false_positives, true_negatives, false_negatives

    @abstractmethod
    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> Metrics:
        """
        Provided tensors associated with the various outcomes from predictions compared to targets in the form of
        true positives, false positives, true negatives, and false negatives, returns a dictionary of Scalar metrics.
        For example, one might compute recall as true_positives/(true_positives + false_negatives). The shape of these
        tensors is likely specific to the kind of classification being done (multi-class vs. binary) and the way
        predictions and targets have been provided etc.

        Args:
            true_positives (torch.Tensor): Counts associated with positive predictions of a class and true positives
                for that class
            false_positives (torch.Tensor): Counts associated with positive predictions of a class and true negatives
                for that class
            true_negatives (torch.Tensor): Counts associated with negative predictions of a class and true negatives
                for that class
            false_negatives (torch.Tensor): Counts associated with negative predictions of a class and true positives
                for that class

        Raises:
            NotImplementedError: Must be implemented by the inheriting class

        Returns:
            Metrics: Metrics computed from the provided outcome counts
        """
        raise NotImplementedError


class BinaryClassificationMetric(ClassificationMetric):
    def __init__(
        self,
        name: str,
        label_dim: int | None = None,
        batch_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        pos_label: int = 1,
        threshold: float | int | None = None,
        discard: set[MetricOutcome] | None = None,
    ) -> None:
        """
        A Base class for BINARY classification metrics that can be computed using the true positives (tp),
        false positive (fp), false negative (fn) and true negative (tn) counts. These counts are computed for
        each class independently. How they are composed together for the metric is left to inheriting classes.

        On each update, the true_positives, false_positives, false_negatives and true_negatives counts for the
        provided predictions and targets are accumulated into ``self.true_positives``, ``self.false_positives``,
        ``self.false_negatives`` and ``self.true_negatives``, respectively, for each label type. This reduces the
        memory footprint required to compute metrics across  rounds. The user needs to define the
        ``compute_from_counts`` method which returns a dictionary of Scalar metrics given the true_positives,
        false_positives, false_negatives, and true_negatives counts. The accumulated counts are reset by the ``clear``
        method. If your subclass returns multiple metrics you may need to also override the `__call__` method.

        If the predictions provided are continuous in value, then the associated counts will also be continuous
        ("soft"). For example, with a target of 1, a prediction of 0.8 contributes 0.8 to the true_positives count and
        0.2 to the false_negatives.

        NOTE: For this class, the predictions and targets passed to the update function MUST have the same shape

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        that argument to be as such.

        Args:
            name (str): The name of the metric.
            label_dim (int | None, optional): Specifies which dimension in the provided tensors corresponds to the
                label dimension. During metric computation, this dimension must have size of AT MOST 2. If left as
                None, this class will assume that each entry in the tensor corresponds to a prediction/target, with
                the positive class indicated by predictions of 1. Defaults to None.
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample INDIVIDUALLY. For example, if batch_dim = 1 and label_dim = 0, then

                .. code-block:: python

                    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])

                    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

                    self.tp = torch.Tensor([[0], [4]])

                    self.tn = torch.Tensor([[2], [0]])

                    self.fp = torch.Tensor([[1], [0]])

                    self.fn = torch.Tensor([[1], [0]])

                NOTE: The resulting counts will always be presented batch dimension first, then label dimension,
                regardless of input shape. Defaults to None.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to float32
            pos_label (int, optional): The label relative to which to report the counts. Must be either 0 or 1.
                Defaults to 1.
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Value of None leaves preds unchanged.
                Defaults to None.
            discard (set[MetricOutcome] | None, optional): One or several of MetricOutcome values. Specified outcome
                counts will not be accumulated. Their associated attribute will remain as an empty pytorch tensor.
                Useful for reducing the memory footprint of metrics that do not use all of the counts in their
                computation. Defaults to None.
        """
        super().__init__(
            name=name,
            dtype=dtype,
            label_dim=label_dim,
            batch_dim=batch_dim,
            threshold=threshold,
            discard=discard,
        )
        assert pos_label == 0 or pos_label == 1, "pos_label must be either 0 or 1"
        self.pos_label = pos_label

    def _postprocess_count_tensor(self, count_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a count tensor, in the various forms that it might appear, we need to post process these so that they
        can be returned to the user in the appropriate format. The structure of the counts tensors after processing
        a batch of data will differ depending on whether the label_dim and batch_dim have been specified.

        NOTE: If the count has been specified as discarded, it will always simply be an empty tensor

        - If both label_dim and batch_dim have been provided, the count tensor will be 2D with the batch dimension
          first, followed by the label dimension.
        - If batch_dim has been provided, but not label_dim, the count tensor will be 1D with a single count
          associated with each sample in the batch.
        - If label_dim has been provided, but not batch_dim, the count tensor will have 1 or 2 elements, because it's
          a binary problem.
        - If neither has been provided then the tensor will have 1 element corresponding to the count over all
          tensor elements.

        Args:
            count_tensor (torch.Tensor): Count tensor with the correct shape and meaning.

        Raises:
            ValueError: Raises errors if the tensor does not have the right shape for the expected setting

        Returns:
            torch.Tensor: Count tensor of the appropriate shape and structure.
        """
        # If tensor is empty, we do nothing
        if count_tensor.numel() == 0:
            return count_tensor

        count_ndims = count_tensor.ndim

        if self.batch_dim is not None and self.label_dim is not None:
            # Both a batch and label dim have been specified. So tensor should be 2D and either have 1 or 2 columns
            assert (
                count_ndims == 2
            ), f"Batch and label dims have been specified, tensor should be 2D, but got {count_ndims}"

            if self.batch_dim > self.label_dim:
                # reshape so that batch dimension comes first
                count_tensor = count_tensor.transpose(0, 1)

            if count_tensor.shape[1] == 2:
                # Always return the class with label 1 (pos_label 0 will be handled by rearranging counts if necessary)
                return count_tensor[:, 1].unsqueeze(1)
            elif count_tensor.shape[1] == 1:
                return count_tensor
            else:
                raise ValueError(f"Label dimension has unexpected size of {count_tensor.shape[1]}")
        elif self.batch_dim is not None:
            # Batch dim has been specified but label dim has not, Tensor should be 1D equivalent to size of the batch
            assert (
                count_ndims == 1
            ), f"Batch dim has been specified but not label dim, tensor should be 1D but got {count_ndims}D"
            return count_tensor.unsqueeze(1)
        else:
            # Tensor should be 1D equivalent to size of labels dimension if it has been specified or a single element
            # if label dims has not be specified. The label dimension is of size at most 2, but can be of size 1 if
            # there is a dimension for an "implied" label (i.e. 0.8 representing vector predictions [0.2, 0.8]). If
            # there is no label dimension specified, there should only be a single element as well.
            assert (
                count_ndims <= 1
            ), f"Batch dim has not been specified, tensor should be 0 or 1D but got {count_ndims}"
            if count_tensor.numel() == 2:
                assert self.label_dim is not None, "self.label_dim is None but got two elements in the count_tensor"
                # Always return the class with label 1
                return count_tensor[1].unsqueeze(0)
            elif count_tensor.numel() == 1:
                return count_tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"Too many elements in the count tensor, expected 2 or less and got {count_tensor.numel()}"
                )

    def count_tp_fp_tn_fn(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
         Given two tensors containing model predictions and targets, returns the number of true positives (tp), false
         positives (fp), true negatives (tn), and false negatives (fn).

        The shape of these counts depends on the values of ``self.batch_dim`` and ``self.label_dim`` are specified.
        They also depend on the shape of the input and target for this class. As binary predictions may be explicit
        (vector encoded) or implicit (single value implying values for the negative and positive classes).

         If any of the true positives, false positives, true negative, or false negative counts were specified to be
         discarded during initialization of the class, then that count will not be computed and an empty tensor will be
         returned in its place.

         Args:
             preds (torch.Tensor): Tensor containing model predictions. Must be the same shape as targets
             targets (torch.Tensor): Tensor containing prediction targets. Must be same shape as preds.

         Returns:
             tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors containing the counts along the
                 specified dimensions for each of true positives, false positives, true negative, and false negative,
                 respectively.
        """
        # Transform preds and targets as necessary/specified before computing counts
        preds, targets = self._transform_tensors(preds, targets)
        # Make sure the tensors have the correct value ranges for sensible computation of the counts
        super()._assert_correct_ranges(preds, targets)

        # Make sure after transformation the preds and targets have the right shape
        assert (
            preds.shape == targets.shape
        ), f"Preds and targets must have the same shape but got {preds.shape} and {targets.shape} respectively."

        # Assert that the label dimension for these tensors is of size 2 at most.
        if self.label_dim is not None:
            assert preds.shape[self.label_dim] <= 2, (
                f"Label dimension for preds tensor is greater than 2 {preds.shape[self.label_dim]}. This class is "
                "meant for binary metric computation only"
            )
            assert targets.shape[self.label_dim] <= 2, (
                f"Label dimension for targets tensor is greater than 2 {targets.shape[self.label_dim]}. This class is "
                "meant for binary metric computation only"
            )

        true_positives, false_positives, true_negatives, false_negatives = super().count_tp_fp_tn_fn(preds, targets)

        true_positives = self._postprocess_count_tensor(true_positives)
        false_positives = self._postprocess_count_tensor(false_positives)
        true_negatives = self._postprocess_count_tensor(true_negatives)
        false_negatives = self._postprocess_count_tensor(false_negatives)

        if self.pos_label == 0:
            # Need to flip the label interpretations
            return true_negatives, false_negatives, true_positives, false_positives

        return true_positives, false_positives, true_negatives, false_negatives

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> Metrics:
        """
        Provided tensors associated with the various outcomes from predictions compared to targets in the form of
        true positives, false positives, true negatives, and false negatives, returns a dictionary of Scalar metrics.
        For example, one might compute recall as true_positives/(true_positives + false_negatives). The shape of these
        tensors are specific to how this object is configured, see class documentation above.

        For this class it is assumed that all counts are presented relative to the class indicated by the `pos_label`
        index. Moreover, they are assumed to either have a single entry or have shape (num_samples, 1). In the former,
        a single count is presented ACROSS all samples relative to the `pos_label` specified. In the latter, counts
        are computed WITHIN each sample, but held separate across samples. A concrete setting where this makes sense
        is binary image segmentation. You can have such counts summed for all pixels within an image, but separate
        per image. A metric could then be computed for each image and then averaged.

        Args:
            true_positives (torch.Tensor): Counts associated with positive predictions and positive labels
            false_positives (torch.Tensor): Counts associated with positive predictions and negative labels
            true_negatives (torch.Tensor): Counts associated with negative predictions and negative labels
            false_negatives (torch.Tensor): Counts associated with negative predictions and positive labels

        Raises:
            NotImplementedError: Must be implemented by the inheriting class

        Returns:
            Metrics: Metrics computed from the provided outcome counts
        """
        raise NotImplementedError

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        """
        User defined convenience method that calculates the desired metric given the predictions and target without
        accumulating it into the class itself

        Raises:
            NotImplementedError: User must define this method.
        """
        raise NotImplementedError


class MultiClassificationMetric(ClassificationMetric):
    def __init__(
        self,
        name: str,
        label_dim: int,
        batch_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        threshold: float | int | None = None,
        ignore_background: int | None = None,
        discard: set[MetricOutcome] | None = None,
    ) -> None:
        """A Base class for multi-class, multi-label classification metrics that can be computed using the true
        positives (tp), false positive (fp), false negative (fn) and true negative (tn) counts. These counts are
        computed for each class independently. How they are composed together for the metric is left to inheriting
        classes.

        On each update, the true_positives, false_positives, false_negatives and true_negatives counts for the
        provided predictions and targets are accumulated into ``self.true_positives``, ``self.false_positives``,
        ``self.false_negatives`` and ``self.true_negatives``, respectively, for each label type. This reduces the
        memory footprint required to compute metrics across rounds. The user needs to define the
        ``compute_from_counts`` method which returns a dictionary of Scalar metrics given the true_positives,
        false_positives, false_negatives, and true_negatives counts. The accumulated counts are reset by the
        ``clear`` method. If your subclass returns multiple metrics you may need to also override the `__call__`
        method.

        If the predictions provided are continuous in value, then the associated counts will also be continuous
        ("soft"). For example, with a target of 1, a prediction of 0.8 contributes 0.8 to the true_positives count and
        0.2 to the false_negatives.

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        that argument to be as such.

        NOTE: If preds and targets passed to update method have different shapes, or end up with different shapes
        after thresholding, this class will attempt to align the shapes by one-hot-encoding one of the tensors in the
        label dimension, if possible.

        Args:
            name (str): The name of the metric.
            label_dim (int): Specifies which dimension in the provided tensors corresponds to the label
                dimension. During metric computation, this dimension must have size of AT LEAST 2.
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample INDIVIDUALLY. For example, if batch_dim = 1 and label_dim = 0, then

                .. code-block:: python

                    p = torch.tensor([[[1., 1., 1., 0.], [0., 0., 0., 0.]], [[0., 0., 0., 1.], [1., 1., 1., 1.]]])

                    t = torch.tensor([[[1., 1., 0., 0.], [0., 0., 0., 0.]], [[0., 0., 1., 1.], [1., 1., 1., 1.]]])

                    self.tp = torch.Tensor([[2, 1], [0, 4]])

                    self.tn = torch.Tensor([[1, 2], [4, 0]])

                    self.fp = torch.Tensor([[1, 0], [0, 0]])

                    self.fn = torch.Tensor([[0, 1], [0, 0]])

                NOTE: The resulting counts will always be presented batch dimension first, then label dimension,
                regardless of input shape. Defaults to None
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to torch.float32
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Value of None leaves preds unchanged.
                Defaults to None.
            ignore_background (int | None): If specified, the FIRST channel of the specified axis is removed prior to
                computing the counts. Useful for removing background classes. Defaults to None.
            discard (set[MetricOutcome] | None, optional): One or several of MetricOutcome values. Specified outcome
                counts will not be accumulated. Their associated attribute will remain as an empty pytorch tensor.
                Useful for reducing the memory footprint of metrics that do not use all of the counts in their
                computation.
        """
        super().__init__(
            name=name,
            dtype=dtype,
            label_dim=label_dim,
            batch_dim=batch_dim,
            threshold=threshold,
            discard=discard,
        )
        if ignore_background is not None:
            log(
                INFO,
                f"ignore_background has been specified. The first channel of dimension {ignore_background} "
                "will be removed from both predictions and targets",
            )
        self.ignore_background = ignore_background

    @classmethod
    def _remove_background(
        cls, ignore_background: int, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Removes the first element (channel) from the dimension specified in ignore_background for both the preds
        and targets tensors. For example, if preds and targets have shape (64, 10, 10, 3) and ignore_background is 3,
        then after removing the specified background, the shapes will be (64, 10, 10, 2)

        The tensors should have the same shape before and after removing the background.

        Args:
            ignore_background (int): Which dimension should have the first element (channel) removed
            preds (torch.Tensor): predictions tensor
            targets (torch.Tensor): targets tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predictions and targets tensor with the appropriate background removed
        """
        assert (
            preds.shape == targets.shape
        ), f"Preds ({preds.shape}) and targets ({targets.shape}) should have the same shape but do not."

        indices = torch.arange(1, preds.shape[ignore_background], device=preds.device)
        preds = torch.index_select(preds, ignore_background, indices)
        targets = torch.index_select(targets, ignore_background, indices.to(targets.device))
        return preds, targets

    @classmethod
    def _transpose_2d_matrix_unless_empty(cls, matrix: torch.Tensor) -> torch.Tensor:
        """
        Helper function to transpose the provided matrix if it is 2D. This is mainly used to put the batch dimension
        before the label dimension if required. The tensor might be empty if it corresponds to a discarded outcome
        type. If so, it is returned unchanged. Otherwise, we throw an error, as the shape is unexpected.

        Args:
            matrix (torch.Tensor): tensor to be transposed

        Returns:
            torch.Tensor: transposed tensor if 2D, unchanged tensor if empty, and throw error if shape differs from
                those two expected settings
        """
        if matrix.ndim == 2:
            return matrix.transpose(0, 1)
        elif matrix.numel() == 0:
            return matrix
        else:
            raise ValueError(f"Expected tensor to either be 2D or empty but has shape {matrix.shape}")

    def _transform_tensors(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given the predictions and targets this function performs a few possible transformations. The first is to map
        boolean tensors to integers for computation. The second is to potentially threshold the predictions if
        self.threshold is not None. Both are facilitated by the base class.

        Thereafter, we attempt to align the tensors if they aren't already of the same shape. This is done by
        attempting to expand the label encodings for class indices to one-hot vectors. See documentation of
        ``align_pred_and_target_shapes`` for more details.

        As a last possible transformation, the background is removed if self.ignore_background is defined.

        Args:
            preds (torch.Tensor): Predictions tensor
            targets (torch.Tensor): Targets tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Potentially transformed predictions and targets tensors, in that order
        """

        preds, targets = super()._transform_tensors(preds, targets)

        # Attempt to automatically match pred and target shape.
        # This supports any combination of hard/soft, vector/not-vector encoded
        preds, targets = align_pred_and_target_shapes(preds, targets, self.label_dim)

        # Remove the background channel from the axis specified by ignore_background_axis
        if self.ignore_background is not None:
            preds, targets = self._remove_background(self.ignore_background, preds, targets)

        return preds, targets

    def count_tp_fp_tn_fn(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given two tensors containing model predictions and targets, returns the number of true positives (tp), false
        positives (fp), true negatives (tn), and false negatives (fn).

        If any of the true positives, false positives, true negative, or false negative counts were specified to be
        discarded during initialization of the class, then that count will not be computed and an empty tensor will be
        returned in its place.

        For this class, counts will either have shape (num_labels,) or (num_samples, num_labels), depending on if the
        `batch_dim` (former case) is specified or not (latter case).

        Args:
            preds (torch.Tensor): Tensor containing model predictions.
            targets (torch.Tensor): Tensor containing prediction targets.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors containing the counts along the
                specified dimensions for each of true positives, false positives, true negative, and false negative
                respectively.
        """

        # Transform preds and targets as necessary/specified before computing counts
        preds, targets = self._transform_tensors(preds, targets)
        # Make sure the tensors have the correct value ranges for sensible computation of the counts
        super()._assert_correct_ranges(preds, targets)

        # Assert that the label dimension for these tensors is not of size 1. This occurs either when considering
        # binary predictions or when both the preds and targets are label index encoded, which is not admissible for
        # this function
        assert self.label_dim is not None
        assert preds.shape[self.label_dim] > 1, (
            "Label dimension for preds tensor is less than 2. Either your label dimension is a single float value "
            "corresponding to a binary prediction or it is a class label that needs to be vector encoded."
        )
        assert targets.shape[self.label_dim] > 1, (
            "Label dimension for targets tensor is less than 2. Either your label dimension is a single float value "
            "corresponding to a binary prediction or it is a class label that needs to be vector encoded."
        )

        true_positives, false_positives, true_negatives, false_negatives = super().count_tp_fp_tn_fn(preds, targets)

        # If batch dim is larger than label_dim, then we re-order them so that batch_dim comes first after summation
        if self.batch_dim is not None and self.batch_dim > self.label_dim:
            return (
                self._transpose_2d_matrix_unless_empty(true_positives),
                self._transpose_2d_matrix_unless_empty(false_positives),
                self._transpose_2d_matrix_unless_empty(true_negatives),
                self._transpose_2d_matrix_unless_empty(false_negatives),
            )

        return true_positives, false_positives, true_negatives, false_negatives

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> Metrics:
        """
        Provided tensors associated with the various outcomes from predictions compared to targets in the form of
        true positives, false positives, true negatives, and false negatives, returns a dictionary of Scalar metrics.
        For example, one might compute recall as true_positives/(true_positives + false_negatives). The shape of these
        tensors is The shape of these tensors are specific to how this object is configured, see class documentation
        above.

        For this class, counts are assumed to have shape (num_labels,) or (num_samples, num_labels). In the former,
        counts have been aggregated ACROSS samples into single count values for each possible label. In the later,
        counts have been aggregated WITHIN each sample and remain separate across examples. A concrete setting where
        this makes sense is image segmentation. You can have such counts summed for all pixels within an image, but
        separate per image. A metric could then be computed for each image and then averaged.

        NOTE: A user can implement further reduction along the label dimension (summing TPs across labels for example),
        if desired. It just needs to be handled in the implementation of this function

        Args:
            true_positives (torch.Tensor): Counts associated with positive predictions of a class and true positives
                for that class
            false_positives (torch.Tensor): Counts associated with positive predictions of a class and true negatives
                for that class
            true_negatives (torch.Tensor): Counts associated with negative predictions of a class and true negatives
                for that class
            false_negatives (torch.Tensor): Counts associated with negative predictions of a class and true positives
                for that class

        Raises:
            NotImplementedError: Must be implemented by the inheriting class

        Returns:
            Metrics: Metrics computed from the provided outcome counts
        """
        raise NotImplementedError

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        """
        User defined convenience method that calculates the desired metric given the predictions and target without
        accumulating it into the class itself

        Raises:
            NotImplementedError: User must define this method.
        """
        raise NotImplementedError
