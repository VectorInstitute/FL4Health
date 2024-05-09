from typing import Optional, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.sequential_split_models import SequentiallySplitModel


class MoonModel(SequentiallySplitModel):
    def __init__(
        self, base_module: nn.Module, head_module: nn.Module, projection_module: Optional[nn.Module] = None
    ) -> None:
        """
        A MoonModel is a specific type of sequentially split model, where one may specify an optional projection
        module to be used for feature manipulation. The model always stores the features produced by the base module
        as they will be used in contrastive loss function calculations. These features are, also, always flattened to
        be compatible with such losses.

        Args:
            base_module (nn.Module): Feature extractor component of the model
            head_module (nn.Module): Classification (or other type) of head used by the model
            projection_module (Optional[nn.Module], optional): An optional module for manipulating the features before
                they are passed to the head_module. Defaults to None.
        """

        # Features are forced to be stored and flattened in this model, as it is expected to always be used with the
        # contrastive loss function.
        super().__init__(base_module, head_module, flatten_features=True)
        self.projection_module = projection_module

    def sequential_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overriding the sequential forward of the SequentiallySplitModel parent to allow for the injection of a
        projection module into the forward pass. The remainder of the functionality stays the same. That is,
        We run a forward pass using the sequentially split modules base_module -> head_module.

        Args:
            input (torch.Tensor): Input to the model forward pass. Expected to be of shape (batch_size, *)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the predictions and features tensor from the sequential forward
        """
        x = self.base_module.forward(input)
        # A projection module is optionally specified for MOON models. If no module is provided, it is simply skipped
        features = self.projection_module.forward(x) if self.projection_module else x
        predictions = self.head_module.forward(features)
        return predictions, features
