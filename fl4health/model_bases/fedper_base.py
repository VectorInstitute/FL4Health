from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class FedPerModel(PartialLayerExchangeModel):
    def __init__(
        self, global_feature_extractor: nn.Module, local_prediction_head: nn.Module, flatten_features: bool = False
    ) -> None:
        """
        Implementation of the FedPer model structure: https://arxiv.org/pdf/1912.00818.pdf
        The architecture is fairly straightforward. The global module represents the first set of layers. These are
        learned with FedAvg. The local_prediction_head are the last layers, these are not exchanged with the server.
        The approach resembles FENDA, but vertical rather than parallel models.

        NOTE: The structure is similar to MOON but only a subset of the components are aggregated server-side.

        Args:
            global_feature_extractor (nn.Module): First set of layers. These are exchanged with the server.
            local_prediction_head (nn.Module): Final set of layers. These are not aggregated by the server.
            flatten_features (bool): Whether or not the forward should flatten the produced features across the batch.
                Defaults to False.
        """
        super().__init__()
        self.global_feature_extractor = global_feature_extractor
        self.local_prediction_head = local_prediction_head
        self.flatten_features = flatten_features

    def layers_to_exchange(self) -> List[str]:
        return [
            layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("global_feature_extractor.")
        ]

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        features = self.global_feature_extractor.forward(input)
        preds = {"prediction": self.local_prediction_head.forward(features)}
        features = {"features": features} if not self.flatten_features else {features.reshape(len(features), -1)}
        # Return preds and features as separate dictionary
        return preds, features
