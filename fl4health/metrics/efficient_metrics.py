from logging import WARNING

import torch
from flwr.common.logger import log
from flwr.common.typing import Metrics, Scalar

from fl4health.metrics.efficient_metrics_base import BinaryClassificationMetric


def compute_dice_on_count_tensors(
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    zero_division: float | None,
) -> torch.Tensor:
    """
    Given a set of count tensors representing true positives (TP), false positives (FP), and false negatives (FN),
    compute the  Dice score as 2*TP/(2*TP + FP + FN) ELEMENTWISE. The zero division argument determines how to deal
    with examples with all true negatives, which implies that TP + FP + FN = 0 and an undefined value.

    Args:
        true_positives (torch.Tensor): count of true positives in each entry
        false_positives (torch.Tensor): count of false positives in each entry
        false_negatives (torch.Tensor): count of false negatives in each entry
        zero_division (float | None): How to deal with zero division. If None, the values with zero division are
            simply dropped. If a float is specified, this value is injected into each Dice score that would have
            been undefined.

    Returns:
        torch.Tensor: Dice scores computed for each element in the TP, FP, FN tensors computed ELEMENTWISE with
            replacement or dropping of undefined entries. The tensor returned is flattened to be 1D.
    """
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
        where a single predictions is a 2D vector [0.2, 0.8] or a float 0.8 (with the complement implied)

        NOTE: For this class, the predictions and targets passed to the update function MUST have the same shape

        NOTE: The default behavior for Dice Scores is to compute the mean over each SAMPLE of the dataset being
        measured. In the image domain, for example, this means that the Dice score is computed for each image
        separately and then averaged across images (then classes) to produce a single score. This is accomplished
        by specifying the batch_dim here. If, however, you would like to compute the Dice score over ALL TP, FP, FNs
        across all samples (then classes) as a single count, batch_dim = None is appropriate.

        NOTE: Preds and targets are expected to have elements in the interval [0, 1] or to be thresholded, using
        the argument of this class to be as such.

        Args:
            batch_dim (int | None, optional): If None, then counts are aggregated across the batch dimension. If
                specified, counts will be computed along the dimension specified. That is, counts are maintained for
                each training sample INDIVIDUALLY. For example, if batch_dim = 1 and label_dim = 0, then

                .. code-block:: python

                    predictions = torch.tensor([[[0, 0, 0, 1], [1, 1, 1, 1]]]) # shape (1, 2, 4)

                    targets = torch.tensor([[[0, 0, 1, 0], [1, 1, 1, 1]]]) # shape (1, 2, 4)

                    self.true_positives = torch.Tensor([[0], [4]])

                    self.true_negatives = torch.Tensor([[2], [0]])

                    self.false_positives = torch.Tensor([[1], [0]])

                    self.false_negatives = torch.Tensor([[1], [0]])

                In computing the Dice score, we get scores for each sample [[2*0/(2*0 +1+1)], [2*4/(2*4+0+0)]]. These
                are then averaged to get 0.5.
            name (str): Name of the metric. Defaults to 'BinaryDice'
            label_dim (int | None, optional): Specifies which dimension in the provided tensors corresponds to the
                label dimension. During metric computation, this dimension must have size of AT MOST 2. If left as
                None, this class will assume that each entry in the tensor corresponds to a prediction/target, with
                the positive class indicated by predictions of 1. Defaults to None.
            dtype (torch.dtype): The dtype to store the counts as. If preds or targets can be continuous, specify a
                float type. Otherwise specify an integer type to prevent overflow. Defaults to torch.float32
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
        super().__init__(
            name=name,
            batch_dim=batch_dim,
            label_dim=label_dim,
            dtype=dtype,
            threshold=threshold,
            pos_label=pos_label,
            discard=None,
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
