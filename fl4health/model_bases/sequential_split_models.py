from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class SequentiallySplitModel(nn.Module):
    def __init__(self, base_module: nn.Module, head_module: nn.Module) -> None:
        """
        These models are split into two sequential stages. The first is a feature extractor module. The second is the
        head_module. Features are extracted from the head module and stored for later use, if required

        Args:
            base_module (nn.Module): Feature extraction module
            head_module (nn.Module): Classification (or other type) of head that acts on the output from the base
                module
        """
        super().__init__()
        self.base_module = base_module
        self.head_module = head_module

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        features = self.base_module.forward(input)
        predictions = {"prediction": self.head_module.forward(features)}
        # Return preds and a features dictionary representing the output of the base_module
        return predictions, {"features": features}


class SequentiallySplitExchangeBaseModel(SequentiallySplitModel, PartialLayerExchangeModel):
    """
    This model is a specific type of sequentially split model, where we specify the layers to be exchanged as being
    the base_module.
    """

    def layers_to_exchange(self) -> List[str]:
        return [layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("base_module.")]
