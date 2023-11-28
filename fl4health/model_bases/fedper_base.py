from typing import List

import torch.nn as nn

from fl4health.model_bases.moon_base import MoonModel
from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class FedPerModel(MoonModel, PartialLayerExchangeModel):
    def __init__(
        self, global_feature_extractor: nn.Module, local_prediction_head: nn.Module, flatten_features: bool = False
    ) -> None:
        """
        Implementation of the FedPer model structure: https://arxiv.org/pdf/1912.00818.pdf
        The architecture is fairly straightforward. The global module represents the first set of layers. These are
        learned with FedAvg. The local_prediction_head are the last layers, these are not exchanged with the server.
        The approach resembles FENDA, but vertical rather than parallel models. It also resembles MOON, but with
        partial weight exchange for weight aggregation. Hence, it inherits from MoonModel and PartialLayerExchangeModel

        Args:
            global_feature_extractor (nn.Module): First set of layers. These are exchanged with the server.
            local_prediction_head (nn.Module): Final set of layers. These are not aggregated by the server.
            flatten_features (bool): Whether or not the forward should flatten the produced features across the batch.
                Flattening of the features can be used to ensure that the features produced are compatible with the
                MOON-based constrative loss functions. This allows a FedPer model to be used with a MOON client.
                Defaults to False.
        """
        super().__init__(
            base_module=global_feature_extractor, head_module=local_prediction_head, projection_module=None
        )
        # This overrides the flatten features option if we don't want to flatten them
        # (i.e. not using contrastive loss) in the Moon Client
        self.flatten_features = flatten_features

    def layers_to_exchange(self) -> List[str]:
        return [layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("base_model.")]
