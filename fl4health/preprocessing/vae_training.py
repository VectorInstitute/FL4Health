from typing import Tuple

import numpy as np
import torch


class AETransformer:
    """Transform function to replace the target in the dataset with data for
    self-supervised approaches like VAEs. For a conditional model, it can concatenate
    the condition with the data sample.
    """

    def __init__(self, condition: str = "") -> None:
        """Initializes the AETransformer with an optional condition.

        Args:
            condition (str, optional): Condition for the transformation. Defaults to "".
        """
        self.condition = condition

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
        if self.condition == "label":
            return torch.from_numpy(np.concatenate((sample, target), axis=None)), torch.from_numpy(sample)
        elif self.condition.isdigit():
            # Custom condition from the client
            return torch.from_numpy(
                np.concatenate((sample, torch.tensor(int(self.condition)).numpy()), axis=None)
            ), torch.from_numpy(sample)
        else:
            # Not conditional
            return torch.from_numpy(sample), torch.from_numpy(sample)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
