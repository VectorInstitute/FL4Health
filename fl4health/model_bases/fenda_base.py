from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from fl4health.model_bases.parallel_split_models import ParallelSplitHeadModule, ParallelSplitModel
from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class FendaModel(PartialLayerExchangeModel, ParallelSplitModel):
    def __init__(self, local_module: nn.Module, global_module: nn.Module, model_head: ParallelSplitHeadModule) -> None:
        ParallelSplitModel.__init__(
            self, first_module=local_module, second_module=global_module, model_head=model_head
        )

    def layers_to_exchange(self) -> List[str]:
        return [layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("second_module.")]


class FendaModelWithFeatureState(FendaModel):
    def __init__(
        self,
        local_module: nn.Module,
        global_module: nn.Module,
        model_head: ParallelSplitHeadModule,
        flatten_features: bool = False,
    ) -> None:
        super().__init__(local_module, global_module, model_head)
        self.flatten_features = flatten_features

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # input is expected to be of shape (batch_size, *)
        local_output = self.first_module.forward(input)
        global_output = self.second_module.forward(input)
        preds = {"prediction": self.model_head.forward(local_output, global_output)}
        features = {
            "local_features": local_output.reshape(len(local_output), -1) if self.flatten_features else local_output,
            "global_features": (
                global_output.reshape(len(global_output), -1) if self.flatten_features else global_output
            ),
        }
        # Return preds and features as separate dictionary as in moon base
        return preds, features
