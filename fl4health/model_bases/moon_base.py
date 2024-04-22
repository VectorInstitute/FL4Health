from typing import Dict, Optional, Tuple

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

        # Features are forced to be stored in this model, as it is expected to always be used with the contrastive
        # loss function.
        super().__init__(base_module, head_module)
        self.projection_module = projection_module

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        x = self.base_module.forward(input)
        features = self.projection_module.forward(x) if self.projection_module else x
        predictions = {"prediction": self.head_module.forward(features)}

        # Return preds and features
        # Features to be returned are always flattened to be compatible with contrastive loss calculations
        return predictions, {"features": features.reshape(len(features), -1)}
