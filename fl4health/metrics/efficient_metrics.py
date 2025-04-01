from logging import WARNING

import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics, Scalar

from fl4health.metrics.efficient_metrics_base import (
    BinaryClassificationMetric,
    MetricOutcome,
    MultiClassificationMetric,
)


def compute_dice_on_count_tensors(
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    zero_division: float | None,
) -> torch.Tensor:
    # Compute union and intersection
    numerator = 2 * true_positives  # Equivalent to 2 times the intersection
    denominator = 2 * true_positives + false_positives + false_negatives  # Equivalent to the union

    # Remove or replace dice score that will be null due to zero division
    if zero_division is None:
        numerator = numerator[denominator != 0]
        denominator = denominator[denominator != 0]
    else:
        numerator[denominator == 0] = zero_division
        denominator[denominator == 0] = 1

    # Return individual dice coefficients
    return numerator / denominator


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
        Computes the Mean DICE Coefficient between class predictions and targets with multiple classes.

        NOTE: The default behavior for Dice Scores is to compute the mean over each sample of the dataset being
        measured. In the image domain, for example, this means that the Dice score is computed for each image
        separately and then averaged across images (then classes) to produce a single score. This is accomplished
        by specifying your batch_dim here. If, however, you would like to compute the Dice score over ALL TP, FP, FNs
        across all images (then classes) as a single count, batch_dim = None is appropriate. These two notions
        are equivalent if all images are the same size, but ARE NOT the same if they differ.

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        that argument to be as such.

        NOTE: If preds and targets passed to update method have different shapes, this class will attempt to align the
        shapes by one-hot-encoding one of the tensors if possible.

        NOTE: In the case of BINARY predictions/targets with 2 channels, the result will be the AVERAGE of the Dice
        score for the two channels. If you want single scores associated with one of the binary labels, use
        BinaryDice.

        Args:
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample INDIVIDUALLY. For example, if batch_dim = 1 and label_dim = 0, then

                .. code-block:: python

                .. code-block:: python

                    p = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]])

                    t = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]])

                    self.tp = torch.Tensor([[0], [4]])

                    self.tn = torch.Tensor([[2], [0]])

                    self.fp = torch.Tensor([[1], [0]])

                    self.fn = torch.Tensor([[1], [0]])

                NOTE: The resulting counts will always be presented batch dimension first, then label dimension,
                regardless of input shape.
            label_dim (int): Specifies which dimension in the provided tensors corresponds to the label
                dimension. During metric computation, this dimension must have size of AT LEAST 2.
            name (str): Name of the metric. Defaults to 'MultiClassDice'
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to torch.float32
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Default of None leaves preds unchanged.
            ignore_background (int | None): If specified, the FIRST channel of the specified axis is removed prior to
                computing the counts. Useful for removing background classes. Defaults to None.
            zero_division (float | None, optional): Set what the individual dice coefficients should be when there is
                a zero division (only true negatives present). How this argument affects the final DICE score will vary
                depending on the DICE scores for other labels. If left as None, the resultant dice coefficients will
                be excluded from the average/final dice score.
        """
        super().__init__(
            name=name,
            batch_dim=batch_dim,
            label_dim=label_dim,
            dtype=dtype,
            threshold=threshold,
            ignore_background=ignore_background,
            discard={MetricOutcome.TRUE_NEGATIVE},
        )
        self.zero_division = zero_division

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        false_negatives: torch.Tensor,
        true_negatives: torch.Tensor,
    ) -> Metrics:
        # compute dice coefficients and return mean
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return {self.name: torch.mean(dice).item()}

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        true_positives, false_positives, false_negatives, _ = self.count_tp_fp_fn_tn(input, target)
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return torch.mean(dice).item()


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
        Computes the DICE Coefficient between binary predictions and targets. These can be vector encoded or
        just continuous predictions with an implicit positive class.

        NOTE: The default behavior for Dice Scores is to compute the mean over each sample of the dataset being
        measured. In the image domain, for example, this means that the Dice score is computed for each image
        separately and then averaged across images (then classes) to produce a single score. This is accomplished
        by specifying your batch_dim here. If, however, you would like to compute the Dice score over ALL TP, FP, FNs
        across all images (then classes) as a single count, batch_dim = None is appropriate. These two notions
        are equivalent if all images are the same size, but ARE NOT the same if they differ.

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        that argument to be as such.

        NOTE: For this class, the predictions and targets passed to the update function MUST have the same shape

        Args:
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
                regardless of input shape.
            name (str): Name of the metric. Defaults to 'BinaryDice'
            label_dim (int | None, optional): Specifies which dimension in the provided tensors corresponds to the
                label dimension. During metric computation, this dimension must have size of AT MOST 2. If left as
                None, this class will assume that each entry in the tensor corresponds to a prediction or target.
                Defaults to None.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to torch.float32
            pos_label (int, optional): The label relative to which to report the counts. Must be either 0 or 1.
                Defaults to 1.
            threshold (float | int | None, optional): A float for thresholding values or an integer specifying the
                index of the label dimension. If a float is given, predictions below the threshold are mapped
                to 0 and above are mapped to 1. If an integer is given, predictions are binarized based on the class
                with the highest prediction where the specified axis is assumed to contain a prediction for each class
                (where its index along that dimension is the class label). Default of None leaves preds unchanged.
                Defaults to None.
            zero_division (float | None, optional): Set what the individual dice coefficients should be when there is
                a zero division (only true negatives present). If None, these examples will be dropped. If all
                components are only TNs, then NaN will be returned.
        """
        super().__init__(
            name=name,
            batch_dim=batch_dim,
            label_dim=label_dim,
            dtype=dtype,
            threshold=threshold,
            pos_label=pos_label,
            discard={MetricOutcome.TRUE_NEGATIVE},
        )
        self.zero_division = zero_division

    def compute_from_counts(
        self,
        true_positives: torch.Tensor,
        false_positives: torch.Tensor,
        false_negatives: torch.Tensor,
        true_negatives: torch.Tensor,
    ) -> Metrics:
        # compute dice coefficients and return mean
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return {self.name: torch.mean(dice).item()}

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> Scalar:
        true_positives, false_positives, false_negatives, _ = self.count_tp_fp_fn_tn(input, target)
        dice = compute_dice_on_count_tensors(true_positives, false_positives, false_negatives, self.zero_division)
        if dice.numel() == 0:
            log(WARNING, "Currently, Dice score is undefined due to only true negatives present")
        return torch.mean(dice).item()
