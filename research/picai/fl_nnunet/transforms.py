from logging import WARNING

import torch
from flwr.common.logger import log


log(
    WARNING,
    (
        "You are using the old version of nnunet transforms from the research folder. "
        "These methods have been moved to nnunet_utils in fl4health.utils"
    ),
)


# Some transform functions to use with the TransformsMetric Class
def get_annotations_from_probs(preds: torch.Tensor, has_regions: bool = False, threshold: float = 0.5) -> torch.Tensor:
    """
    Converts the model output probabilities to predicted annotations.

    Args:
        preds (torch.Tensor): The one hot encoded model output probabilities with shape (batch, classes,
            *additional_dims). The background should be a separate class.
        has_regions (bool, optional): If True, predicted annotations can be multiple classes at once. The exception
            is the background class which is assumed to be the first class (class 0). If False, each value in
            predicted annotations has only a single class. Defaults to False.
        threshold (float): When has_regions is True, this is the threshold value used to determine whether or not an
            output is a part of a class.

    Returns:
        (torch.Tensor): tensor containing the predicted annotations as a one hot encoded binary tensor of 64-bit
            integers.
    """
    if has_regions:
        pred_annotations = preds > threshold
        # Mask is the inverse of the background class. Ensures that values
        # classified as background are not part of another class
        mask = ~pred_annotations[:, 0]
        return pred_annotations * mask
    pred_annotations = preds.argmax(1)[:, None]  # shape (batch, 1, additional_dims)
    # one hot encode (OHE) predicted annotations again
    # WARNING: Note the '_' after scatter. scatter_ and scatter are both
    # functions with different functionality. It is easy to introduce a bug
    # here by using the wrong one
    pred_annotations_one_hot = torch.zeros(preds.shape, device=preds.device, dtype=torch.float32)
    pred_annotations_one_hot.scatter_(1, pred_annotations, 1)  # ohe -> One Hot Encoded
    # convert output preds to long since it is binary
    return pred_annotations_one_hot.long()


def collapse_one_hot_tensor(input: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Collapses a one hot encoded tensor so that they are no longer one hot encoded.

    Args:
        input (torch.Tensor): The binary one hot encoded tensor.
        dim (int, optional): Dimension over which to collapse the one-hot tensor. Defaults to 0.

    Returns:
        (torch.Tensor): Integer tensor with the specified dim collapsed.
    """
    return torch.argmax(input.long(), dim=dim).to(input.device)
