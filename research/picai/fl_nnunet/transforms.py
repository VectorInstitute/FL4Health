from typing import Tuple

import torch

# Some transform functions to use with the TransformsMetric Class


def get_prob_from_logits(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the model output logits to probabilities

    Args:
        preds (torch.Tensor): The one hot encoded model output logits with
            shape (batch, classes, *additional_dims)
        targets (torch.Tensor): The targets to evaluate the predictons with.
            The targets are not modified by this function

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is a
            tensor containing the model output probabilities and the second
            element is a tensor containing the unchanged targets
    """
    return torch.sigmoid(preds), targets


def get_ann_from_probs(
    preds: torch.Tensor, targets: torch.Tensor, has_regions: bool = False, threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts the model output probabilities to predicted annotations

    Args:
        preds (torch.Tensor): The one hot encoded model output probabilities
            with shape (batch, classes, *additional_dims). The background should be a seperate class
        targets (torch.Tensor): The targets to evaluate the predictions with.
            The targets are not modified by this function
        has_regions (bool, optional): If True, predicted annotations can be
            multiple classes at once. The exception is the background class
            which is assumed to be the first class (class 0). If False, each
            value in predicted annotations has only a single class. Defaults to
            False.
        threshold (float): When has_regions is True, this is the threshold
            value used to determine whether or not an output is a part of a
            class

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is a
            tensor containing the predicted annotations and the second element
            is a tensor containing the unchanged targets. The predicted
            annotations as a one hot encoded binary tensor of 64-bit integers
    """
    if has_regions:
        pred_ann = preds > threshold
        # Mask is the inverse of the background class. Ensures that values
        # classified as background are not part of another class
        mask = ~pred_ann[:, 0]
        return pred_ann * mask, targets
    else:
        pred_ann = preds.argmax(1)[:, None]  # shape (batch, 1, additional_dims)
        # one hot encode predicted annotations again
        pred_ann_ohe = torch.zeros(preds.shape, device=preds.device, dtype=torch.float32)
        pred_ann_ohe.scatter_(1, pred_ann, 1)
        # convert output preds to long since it is binary
        return pred_ann_ohe.long(), targets


def collapse_ohe_target(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapses the targets so that they are no longer one hot encoded

    Args:
        preds (torch.Tensor): The model predictions. Not modified by this
            function
        targets (torch.Tensor): The one hot encoded targets with shape
            (batch, classes, *additional_dims)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element is a
            tensor with the unchanged preds and the second element is a tensor
            with the output targets with shape (batch, *additional_dims).
    """
    return preds, torch.argmax(targets.long(), dim=1).to(targets.device)
