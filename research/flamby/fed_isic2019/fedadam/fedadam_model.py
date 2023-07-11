from logging import INFO

import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import Baseline
from flwr.common.logger import log


class FedAdamEfficientNet(nn.Module):
    """FedAdam implements server-side momentum in aggregating the updates from each client. For layers that carry state
    that must remain non-negative, like BatchNormalization layers (present in EffcientNet), they may become negative
    due to momentum carrying updates past the origin. For Batch Normalization this means that the variance state
    estimated during training and applied during evaluation may become negative. This blows up the model. In order
    to get around this issue, we modify all batch normalization layers in EfficientNet to not carry such state by
    setting track_running_stats to false.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Baseline()
        # Freeze layers to reduce trainable parameters.
        self.modify_batch_normalization_layers()

    def modify_batch_normalization_layers(self) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate throught the blocks freezing the specified number up to 15 (all of them)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                log(INFO, f"Modifying Batch Normalization Layer: {name}")
                module.track_running_stats = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
