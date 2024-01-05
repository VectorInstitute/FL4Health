from typing import Tuple

import numpy as np
import torch


class AETransformer:
    """Transform function to replace the target in the dataset with data for
    self-supervised approaches like VAEs. For a conditional model, it can concatenate
    the condition with the data sample.
    """

    def __init__(self, condition: str = "", img_dims: int = 1) -> None:
        """Initializes the AETransformer with an optional condition.

        Args:
            condition (str, optional): Condition for the transformation. Defaults to "".
        """
        self.condition = condition
        self.img_dims = img_dims

    def __call__(self, sample: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transformation to the input data.

        Args:
            sample (np.ndarray): Data sample from the dataset.
            target (Optional[np.ndarray], optional): Target is only necessary
            in CVAEs conditioned on data label. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data format used for training,
            for self-supervised VAE it is (sample, sample). For CVAE conditioned on
            an integer it is ([sample, int], sample). For CVAE conditioned on label,
            it is ([sample, target], sample)
        """
        if self.condition == "":  # Not conditional
            return torch.from_numpy(sample), torch.from_numpy(sample)
        if self.condition == "label":
            condition = torch.tensor(target)
        if self.condition.isdigit():
            condition = torch.tensor(int(self.condition))
        # First match the dimention of sample and condition.
        if self.img_dims > 1:
            # Replicate the condition tensor to match the shape of the matrix tensor.
            expanded_condition = condition.expand_as(torch.from_numpy(sample))
            # Combine the expanded number tensor and the matrix tensor along a new dimension.
            return torch.cat((torch.from_numpy(sample), expanded_condition)), torch.from_numpy(sample)
        return torch.cat((torch.from_numpy(sample), condition.view(-1))), torch.from_numpy(sample)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
