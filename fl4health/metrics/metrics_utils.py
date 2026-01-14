import torch


def compute_dice_on_count_tensors(
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    zero_division: float | None,
) -> torch.Tensor:
    """
    Given a set of count tensors representing true positives (TP), false positives (FP), and false negatives (FN),
    compute the  Dice score as...

    \\[


    \\]
        \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}

    **ELEMENTWISE**. The zero division argument determines how to deal with examples with all true negatives, which
    implies that ``TP + FP + FN = 0`` and an undefined value.

    Args:
        true_positives (torch.Tensor): count of true positives in each entry.
        false_positives (torch.Tensor): count of false positives in each entry.
        false_negatives (torch.Tensor): count of false negatives in each entry.
        zero_division (float | None): How to deal with zero division. If None, the values with zero division are
            simply dropped. If a float is specified, this value is injected into each Dice score that would have
            been undefined.

    Returns:
        (torch.Tensor): Dice scores computed for each element in the TP, FP, FN tensors computed **ELEMENTWISE** with
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


def threshold_tensor(input: torch.Tensor, threshold: float | int) -> torch.Tensor:
    """
    Converts continuous 'soft' tensors into categorical 'hard' ones.

    Args:
        input (torch.Tensor): The tensor to threshold.
        threshold (float | int): A float for thresholding values or an integer specifying the index of the
            label dimension. If a float is given, elements below the threshold are mapped to 0 and above are
            mapped to 1. If an integer is given, elements are thresholded based on the class with the highest
            prediction.

    Returns:
        (torch.Tensor): Thresholded tensor.
    """
    if isinstance(threshold, float):
        thresholded_tensor = torch.zeros_like(input)
        mask_1 = input > threshold
        thresholded_tensor[mask_1] = 1
        return thresholded_tensor
    if isinstance(threshold, int):
        # Use argmax to get predicted class labels (hard_preds) and the one-hot-encode them.
        if threshold >= input.ndim:
            raise ValueError(
                f"Cannot apply argmax to Tensor of shape {input.shape}. "
                f"Label dimension of {threshold} is out of range of tensor with {input.ndim} dimensions."
            )
        hard_input = input.argmax(threshold, keepdim=True)
        input = torch.zeros_like(input)
        input.scatter_(threshold, hard_input, 1)
        return input
    raise ValueError(f"Was expecting threshold argument to be either a float or an int. Got {type(threshold)}")
