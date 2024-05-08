from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class SequentiallySplitModel(nn.Module):
    def __init__(self, base_module: nn.Module, head_module: nn.Module, flatten_features: bool = False) -> None:
        """
        These models are split into two sequential stages. The first is base_module, used as a feature extractor.
        The second is the head_module, used as a classifier. Features are extracted from the base_module and stored
        for later use, if required

        Args:
            base_module (nn.Module): Feature extraction module
            head_module (nn.Module): Classification (or other type) of head that acts on the output from the base
                module
            flatten_features (bool, optional): Whether the feature tensor shapes are to be preserved (false) or if
                they should be flattened to be of shape (batch_size, -1). Flattening may be necessary when using
                certain loss functions, as in MOON, for example. Defaults to False.
        """
        super().__init__()
        self.base_module = base_module
        self.head_module = head_module
        self.flatten_features = flatten_features

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        The features tensor is flattened to be of shape are flattened to be of shape (batch_size, -1). It is expected
        that the feature tensor is BATCH FIRST

        Args:
            features (torch.Tensor): Features tensor to be flattened. It is assumed that this tensor is BATCH FIRST.

        Returns:
            torch.Tensor: Flattened feature tensor of shape (batch_size, -1)
        """
        return features.reshape(len(features), -1)

    def sequential_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a forward pass using the sequentially split modules base_module -> head_module.

        Args:
            input (torch.Tensor): Input to the model forward pass. Expected to be of shape (batch_size, *)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the predictions and features tensor from the sequential forward
        """
        features = self.base_module.forward(input)
        predictions = self.head_module.forward(features)
        return predictions, features

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run a forward pass using the sequentially split modules base_module -> head_module. Features from the
        base_module are stored either in their original shapes are flattened to be of shape (batch_size, -1) depending
        on self.flatten_features

        Args:
            input (torch.Tensor): Input to the model forward pass. Expected to be of shape (batch_size, *)

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Dictionaries of predictions and features
        """
        predictions, features = self.sequential_forward(input)
        predictions_dict = {"prediction": predictions}
        features_dict = (
            {"features": self._flatten_features(features)} if self.flatten_features else {"features": features}
        )
        # Return the prediction dictionary and a features dictionaries representing the output of the base_module
        # either in the standard tensor shape or flattened, to be compatible, for example, with MOON contrastive
        # losses.
        return predictions_dict, features_dict


class SequentiallySplitExchangeBaseModel(SequentiallySplitModel, PartialLayerExchangeModel):
    """
    This model is a specific type of sequentially split model, where we specify the layers to be exchanged as being
    those belonging to the base_module.
    """

    def layers_to_exchange(self) -> List[str]:
        """
        Names of the layers of the model to be exchanged with the server. For these models, we only exchange layers
        associated with the base_model.

        Returns:
            List[str]: The names of the layers to be exchanged with the server. This is used by the FixedLayerExchanger
            class
        """
        return [layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("base_module.")]
