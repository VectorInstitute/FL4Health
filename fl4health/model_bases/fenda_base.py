from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.parallel_split_models import ParallelSplitHeadModule, ParallelSplitModel
from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class FendaModel(PartialLayerExchangeModel, ParallelSplitModel):
    def __init__(self, local_module: nn.Module, global_module: nn.Module, model_head: ParallelSplitHeadModule) -> None:
        """
        This is the base model to be used when implementing FENDA-FL models and training. A FENDA model is essentially
        a parallel split model (i.e. it has two parallel feature extractors), where only one feature extractor is
        exchanged with the server (the global_module) while the other remains local to the client itself.

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


class FendaModelWithFeatureState(FendaModel):
    def __init__(
        self,
        local_module: nn.Module,
        global_module: nn.Module,
        model_head: ParallelSplitHeadModule,
        flatten_features: bool = False,
    ) -> None:
        """
        This is the base model to be used when implementing FENDA-FL models and training when extraction and
        storage of the latent features produced by each of the parallel feature extractors is required/desired. This
        is a FENDA model, but the feature space outputs are guaranteed to be stored with the keys 'local_features'
        and 'global_features' along with the predictions. The user also has the option to "flatten" these features
        to be of shape batch_size x all features

        Args:
            local_module (nn.Module): Feature extraction module that is NOT exchanged with the server
            global_module (nn.Module): Feature extraction module that is exchanged with the server and aggregated with
                other client modules
            model_head (ParallelSplitHeadModule): The model head that takes the output features from both the local
                and global modules to produce a prediction.
            flatten_features (bool, optional): Whether the output features should be flattened to have shape
                batch_size x all features. Defaults to False.
        """
        super().__init__(local_module=local_module, global_module=global_module, model_head=model_head)
        self.flatten_features = flatten_features

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        local_output = self.first_feature_extractor.forward(input)
        global_output = self.second_feature_extractor.forward(input)
        preds = {"prediction": self.model_head.forward(local_output, global_output)}

        if self.flatten_features:
            features = {"local_features": local_output, "global_features": global_output}
        else:
            features = {
                "local_features": local_output.reshape(len(local_output), -1),
                "global_features": global_output.reshape(len(global_output), -1),
            }
        # Return preds and features as separate dictionary as in moon base
        return preds, features
