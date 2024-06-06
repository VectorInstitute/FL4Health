from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.parallel_split_models import ParallelSplitHeadModule, ParallelSplitModel
from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class PerFclModel(PartialLayerExchangeModel, ParallelSplitModel):
    def __init__(self, local_module: nn.Module, global_module: nn.Module, model_head: ParallelSplitHeadModule) -> None:
        """
        Model to be used by PerFCL clients to train models with the PerFCL approach. These models are of type
        ParallelSplitModel and have distinct feature extractors. One of the feature extractors is exchanged with the
        server and aggregated while the other remains local. Each of the extractors produces latent features which
        are flattened and stored with the keys 'local_features' and 'global_features' along with the predictions.

        Args:
            local_module (nn.Module): Feature extraction module that is NOT exchanged with the server
            global_module (nn.Module): Feature extraction module that is exchanged with the server and aggregated with
                other client modules
            model_head (ParallelSplitHeadModule): The model head that takes the output features from both the local
                and global modules to produce a prediction.
        """
        ParallelSplitModel.__init__(
            self, first_feature_extractor=local_module, second_feature_extractor=global_module, model_head=model_head
        )

    def layers_to_exchange(self) -> List[str]:
        return [
            layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("second_feature_extractor.")
        ]

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        local_output = self.first_feature_extractor.forward(input)
        global_output = self.second_feature_extractor.forward(input)
        preds = {"prediction": self.model_head.forward(local_output, global_output)}
        # PerFCL models always store features from the feature extractors
        features = {
            "local_features": local_output.reshape(len(local_output), -1),
            "global_features": global_output.reshape(len(global_output), -1),
        }
        return preds, features
