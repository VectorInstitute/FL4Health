from logging import INFO

import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import Baseline
from flwr.common.logger import log


class FedAdamUNet(nn.Module):
    """FedAdam implements server-side momentum in aggregating the updates from each client. For layers that carry state
    that must remain non-negative, like BatchNormalization layers (present in FedIXI U-Net), they may become negative
    due to momentum carrying updates past the origin. For Batch Normalization this means that the variance state
    estimated during training and applied during evaluation may become negative. This blows up the model. In order
    to get around this issue, we modify all batch normalization layers in the FedIXI U-Net to not carry such state by
    setting track_running_stats to false.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Baseline()
        self.modify_batch_normalization_layers()

    def modify_batch_normalization_layers(self) -> None:
        # Iterate through all named modules of the model and, if we encounter a batch normalization layer, we set
        # track_running_stats to false instead of true.
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm3d):
                log(INFO, f"Modifying Batch Normalization Layer: {name}")
                module.track_running_stats = False
                # NOTE: It's apparently not enough to set this boolean to false. We need to set all of the relevant
                # variable to none, otherwise the layer still tries to apply the stale variables during evaluation
                # leading to eventual NaNs again.
                module.running_mean = None
                module.running_var = None
                module.num_batches_tracked = None
                module.register_buffer("running_mean", None)
                module.register_buffer("running_var", None)
                module.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
