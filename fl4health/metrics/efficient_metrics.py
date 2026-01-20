from logging import WARNING

import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics, Scalar

from fl4health.metrics.efficient_metrics_base import (
    BinaryClassificationMetric,
    ClassificationOutcome,
    MultiClassificationMetric,
)
from fl4health.metrics.metrics_utils import compute_dice_on_count_tensors


class MultiClassDice(MultiClassificationMetric):
    def __init__(
        self,
        batch_dim: int | None,
        label_dim: int,
        name: str = "MultiClassDice",
        dtype: torch.dtype = torch.float32,
        threshold: float | int | None = None,
        ignore_background: int | None = None,
        zero_division: float | None = None,
    ) -> None:
        """
        Computes the Mean Dice Coefficient between class predictions and targets with multiple classes.

        **NOTE**: The default behavior for Dice Scores is to compute the mean over each **SAMPLE** of the dataset being
        measured. In the image domain, for example, this means that the Dice score is computed for each image
        separately and then averaged across images (then classes) to produce a single score. This is accomplished
        by specifying the ``batch_dim`` here. If, however, you would like to compute the Dice score over ALL TP, FP,
        FNs across all samples (then classes) as a single count, ``batch_dim = None`` is appropriate.

        **NOTE**: Preds and targets are expected to have elements in the interval ``[0, 1]`` or to be thresholded,
        using that argument to be as such.

        **NOTE**: If preds and targets passed to the update method have different shapes, this class will attempt to
        align the shapes by one-hot-encoding one (but not both) of the tensors if possible.

        **NOTE**: In the case of **BINARY** predictions/targets with 2 labels, the result will be the **AVERAGE** of
        the Dice score for the two labels. If you want a single score associated with one of the binary labels, use
        ``BinaryDice``.

        Args:
            batch_dim (int | None, optional): If None, the counts along the specified dimension (i.e. for each sample)
                are aggregated and the batch dimension is reduced. If specified, counts will be computed along the
                dimension specified. That is, counts are maintained for each training sample **INDIVIDUALLY**.
                **NOTE**: If ``batch_dim`` is specified, then counts will be presented batch dimension
                first, then label dimension. For example, if ``batch_dim = 1`` and ``label_dim = 0``, then

                ```python

                p = torch.tensor(

                ```
                        [[[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
                    )  # Size([2, 2, 4])

                    t = torch.tensor(
                        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
                    )  # Size([2, 2, 4])

                    self.tp = torch.Tensor([[2, 1], [0, 4]])  # Size([2, 2])

                    self.tn = torch.Tensor([[1, 2], [4, 0]])  # Size([2, 2])

                    self.fp = torch.Tensor([[1, 0], [0, 0]])  # Size([2, 2])

                    self.fn = torch.Tensor([[0, 1], [0, 0]])  # Size([2, 2])

                In computing the Dice score ``(2*tp/(2*tp + fp + fn))``, we get scores for each sample/label pair as
                    ``[[2*2/(2*2+1+0), 2*1/(2*1+0+1)], [0*2/(0*2+0+0), 2*4/(2*4+0+0)]]``.
                Assuming ``zero_division = None``, the undefined calculation at ``(1, 0)`` is dropped and the
                remainder of the individual scores are averaged to be ``(1/3)*(4/5 + 2/3 + 8/8) = 0.8222``.
            label_dim (int): Specifies which dimension in the provided tensors corresponds to the label
                dimension. During metric computation, this dimension must have size of **AT LEAST 2**. Counts are
                always computed along the label dimension. That is, counts are maintained for each output label
                **INDIVIDUALLY**.
            name (str): Name of the metric. Defaults to 'MultiClassDice'
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to ``torch.float32``.
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Default of None leaves preds unchanged.
            ignore_background (int | None): If specified, the **FIRST** channel of the specified axis is removed prior
                to computing the counts. Useful for removing background classes. Defaults to None.
            zero_division (float | None, optional): Set what the individual Dice coefficients should be when there is
                a zero division (only true negatives present). How this argument affects the final Dice score will vary
                depending on the Dice scores for other labels. If left as None, the resultant Dice coefficients will
                be excluded from the average/final Dice score.
        """
        super().__init__(
            name=name,
            batch_dim=batch_dim,
            label_dim=label_dim,
            dtype=dtype,
            threshold=threshold,
            ignore_background=ignore_background,
            discard={ClassificationOutcome.TRUE_NEGATIVE},
        )
        self.zero_division = zero_division

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> Metrics:
        """
        Computes a multi-class Dice score, defined to be the mean Dice score across all labels in the multi-class
        problem. This score is computed relative the outcome counts provided in the form of true positives (TP),
        false positives (FP), and false negatives (FN). Because Dice scores don't factor in true negatives, this
        argument is unused. For a set of counts, the Dice score for a particular label is...

        \\[


        \\]
            \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}.  2*TP/(2*TP + FP + FN).

        For this class, counts are assumed to have shape ``(num_labels,)`` or ``(num_samples, num_labels)``. In the
        former, a single Dice score is computed relative to the counts for each label and then **AVERAGED**. In the
        latter, an **AVERAGE** dice score over both the samples AND labels computed. The second setting is useful, for
        example, if you are computing the Dice score per image and then averaging. The first setting is useful, for
        example, if you want to treat all examples as a **SINGLE** image.

        Args:
            true_positives (torch.Tensor): Counts associated with positive predictions and positive labels.
            false_positives (torch.Tensor): Counts associated with positive predictions and negative labels.
            true_negatives (torch.Tensor): Counts associated with negative predictions and negative labels.
            false_negatives (torch.Tensor): Counts associated with negative predictions and positive labels.

        Returns:
            (Metrics): A mean dice score associated with the counts.
        """
        # compute dice coefficients and return mean
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return {self.name: torch.mean(dice).item()}

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        """
        Computes the Dice score relative to the single input and target tensors provided.

        Args:
            input (torch.Tensor): predictions tensor.
            target (torch.Tensor): target tensor.

        Returns:
            (Scalar): Mean dice score for the provided tensors.
        """
        true_positives, false_positives, true_negatives, false_negatives = self.count_tp_fp_tn_fn(input, target)
        dice_metric = self.compute_from_counts(true_positives, false_positives, true_negatives, false_negatives)
        # Extract the scalar from the dictionary.
        return dice_metric[self.name]


class BinaryDice(BinaryClassificationMetric):
    def __init__(
        self,
        batch_dim: int | None,
        name: str = "BinaryDice",
        label_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        pos_label: int = 1,
        threshold: float | int | None = None,
        zero_division: float | None = None,
    ) -> None:
        """
        Computes the Dice Coefficient between binary predictions and targets. These can be vector encoded or
        just single elements values with an implicit positive class. That is, predictions might be vectorized
        where a single predictions is a 2D vector ``[0.2, 0.8]`` or a float 0.8 (with the complement implied).

        Regardless of how the input is structured, the provided score will be provided with respect to the value of the
        ``pos_label`` variable, which defaults to 1 (and can only have values {0, 1}). That is, the reported score
        will correspond to the score from the perspective of the specified label. For additional documentation see
        that of the parent class ``BinaryClassificationMetric`` and the function ``_post_process_count_tensor``
        therein.

        **NOTE**: For this class, the predictions and targets passed to the update function **MUST** have the same
        shape.

        **NOTE**: The default behavior for Dice Scores is to compute the mean over each **SAMPLE** of the dataset being
        measured. In the image domain, for example, this means that the Dice score is computed for each image
        separately and then averaged across images (then classes) to produce a single score. This is accomplished
        by specifying the ``batch_dim`` here. If, however, you would like to compute the Dice score over ALL TP, FP,
        FNs across all samples (then classes) as a single count, ``batch_dim = None`` is appropriate.

        **NOTE**: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        the argument of this class to be as such.

        Args:
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample **INDIVIDUALLY**. For example, if ``batch_dim = 1`` and ``label_dim = 0``, then

                ```python
                predictions = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])  # shape (1, 2, 4)

                targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])  # shape (1, 2, 4)

                self.true_positives = torch.Tensor([[0], [4]])

                self.true_negatives = torch.Tensor([[2], [0]])

                self.false_positives = torch.Tensor([[1], [0]])

                self.false_negatives = torch.Tensor([[1], [0]])
                ```

                In computing the Dice score, we get scores for each sample ``[[2*0/(2*0 +1+1)]``,
                ``[2*4/(2*4+0+0)]]``. These are then averaged to get 0.5.
            name (str): Name of the metric. Defaults to 'BinaryDice'
            label_dim (int | None, optional): Specifies which dimension in the provided tensors corresponds to the
                label dimension. During metric computation, this dimension must have size of **AT MOST 2**. If left as
                None, this class will assume that each entry in the tensor corresponds to a prediction/target, with
                the positive class indicated by predictions of 1. Defaults to None.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to ``torch.float32``
            pos_label (int, optional): The label relative to which to report the Dice. Must be either 0 or 1.
                Defaults to 1.
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Value of None leaves preds unchanged.
                Defaults to None.
            zero_division (float | None, optional): Set what the individual dice coefficients should be when there is
                a zero division (only true negatives present). If None, these examples will be dropped. If all
                components are only TNs, then NaN will be returned.
        """
        # The right set of counts that can be ignored for Dice computation depends on which label relative to which
        # we're reporting the score. If reporting relative to the positive label, then we need not track
        # True Negatives, as they don't factor into the standard Dice score. On the other hand, if reporting relative
        # to the negative class, we need not keep True Positives around, for the same reason.
        discard = {ClassificationOutcome.TRUE_NEGATIVE} if pos_label == 1 else {ClassificationOutcome.TRUE_POSITIVE}
        super().__init__(
            name=name,
            batch_dim=batch_dim,
            label_dim=label_dim,
            dtype=dtype,
            threshold=threshold,
            pos_label=pos_label,
            discard=discard,
        )
        self.zero_division = zero_division

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        true_negatives: torch.Tensor,
        false_negatives: torch.Tensor,
    ) -> Metrics:
        """
        Computes a binary Dice score associated with the outcome counts provided in the form of true positives (TP),
        false positives (FP), and false negatives (FN). Because Dice scores don't factor in true negatives, this
        argument is unused. For a set of counts, the binary Dice score is...

        \\[


        \\]
            \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}.

        For this class it is assumed that all counts are presented relative to the class indicated by the
        ``pos_label`` index. Moreover, they are assumed to either have a single entry or have shape
        ``(num_samples, 1)``. In the former, a single Dice score is computed relative to the counts. In the latter, a
        **MEAN** dice score over the samples is computed. The second setting is useful, for example, if you are
        computing the Dice score per image and then averaging. The first setting is useful, for example, if you want
        to treat all examples as a **SINGLE** image.

        Args:
            true_positives (torch.Tensor): Counts associated with positive predictions and positive labels.
            false_positives (torch.Tensor): Counts associated with positive predictions and negative labels.
            true_negatives (torch.Tensor): Counts associated with negative predictions and negative labels.
            false_negatives (torch.Tensor): Counts associated with negative predictions and positive labels.

        Returns:
            (Metrics): A mean dice score associated with the counts.
        """
        # compute dice coefficients and return mean
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return {self.name: torch.mean(dice).item()}

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        """
        Computes the Dice score relative to the single input and target tensors provided.

        Args:
            input (torch.Tensor): predictions tensor.
            target (torch.Tensor): target tensor.

        Returns:
            (Scalar): Mean dice score for the provided tensors.
        """
        true_positives, false_positives, true_negatives, false_negatives = self.count_tp_fp_tn_fn(input, target)
        dice_metric = self.compute_from_counts(true_positives, false_positives, true_negatives, false_negatives)
        # Extract the scalar from the dictionary.
        return dice_metric[self.name]
