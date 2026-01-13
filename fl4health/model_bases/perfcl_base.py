import torch
from torch import nn

from fl4health.model_bases.parallel_split_models import ParallelSplitHeadModule, ParallelSplitModel
from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class PerFclModel(PartialLayerExchangeModel, ParallelSplitModel):
    def __init__(self, local_module: nn.Module, global_module: nn.Module, model_head: ParallelSplitHeadModule) -> None:
        """
        Model to be used by PerFCL clients to train models with the PerFCL approach. These models are of type
        ``ParallelSplitModel`` and have distinct feature extractors. One of the feature extractors is exchanged with
        the server and aggregated while the other remains local. Each of the extractors produces latent features which
        are flattened and stored with the keys "local_features" and "global_features" along with the predictions.

        Args:
            local_module (nn.Module): Feature extraction module that is **NOT** exchanged with the server.
            global_module (nn.Module): Feature extraction module that is exchanged with the server and aggregated with
                other client modules.
            model_head (ParallelSplitHeadModule): The model head that takes the output features from both the local
                and global modules to produce a prediction.
        """
        ParallelSplitModel.__init__(
            self, first_feature_extractor=local_module, second_feature_extractor=global_module, model_head=model_head
        )

    def layers_to_exchange(self) -> list[str]:
        """
        Fixes the set of layers to be exchanged with a server. The ``second_feature_extractor`` is assumed to be the
        **GLOBAL** feature extractor for the PerFCL model.

        Returns:
            (list[str]): List of layers associated with the global model (``second_feature_extractor``) corresponding
                to keys in the state dictionary.
        """
        return [layer_name for layer_name in self.state_dict() if layer_name.startswith("second_feature_extractor.")]

    def forward(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Mapping input through the PerFCL model local and global feature extractors and the classification head.

        Args:
            input (torch.Tensor): input is expected to be of shape (``batch_size``, \\*)

        Returns:
            (tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]): Tuple of predictions and feature maps. PerFCL
                predictions are simply stored under the key "prediction." The features for the local and global feature
                extraction modules are stored under keys "local_features" and "global_features," respectively.
        """
        local_output = self.first_feature_extractor.forward(input)
        global_output = self.second_feature_extractor.forward(input)
        preds = {"prediction": self.model_head.forward(local_output, global_output)}
        # PerFCL models always store features from the feature extractors
        features = {
            "local_features": local_output.reshape(len(local_output), -1),
            "global_features": global_output.reshape(len(global_output), -1),
        }
        return preds, features
