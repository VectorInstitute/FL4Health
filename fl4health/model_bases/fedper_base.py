from typing import Dict, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel


class FedPerModel(SequentiallySplitExchangeBaseModel):
    def __init__(
        self, global_feature_extractor: nn.Module, local_prediction_head: nn.Module, flatten_features: bool = False
    ) -> None:
        """
        Implementation of the FedPer model structure: https://arxiv.org/pdf/1912.00818.pdf
        The architecture is fairly straightforward. The global module represents the first set of layers. These are
        learned with FedAvg. The local_prediction_head are the last layers, these are not exchanged with the server.
        The approach resembles FENDA, but vertical rather than parallel models. It also resembles MOON, but with
        partial weight exchange for weight aggregation.

        Args:
            global_feature_extractor (nn.Module): First set of layers. These are exchanged with the server.
            local_prediction_head (nn.Module): Final set of layers. These are not aggregated by the server.
            flatten_features (bool): Whether or not the forward should flatten the produced features across the batch.
                Flattening of the features can be used to ensure that the features produced are compatible with the
                MOON-like contrastive loss functions. Defaults to False.
        """
        super().__init__(base_module=global_feature_extractor, head_module=local_prediction_head)
        self.flatten_features = flatten_features

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        predictions, features = super().forward(input)
        if self.flatten_features:
            features = {"features": features["features"].reshape(len(features), -1)}
        return predictions, features
