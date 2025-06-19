import torch
from flamby.datasets.fed_isic2019 import Baseline
from torch import nn

from research.flamby.utils import shutoff_batch_norm_tracking


class FedAdamEfficientNet(nn.Module):
    """
    FedAdam implements server-side momentum in aggregating the updates from each client. For layers that carry state
    that must remain non-negative, like BatchNormalization layers (present in EfficientNet), they may become negative
    due to momentum carrying updates past the origin. For Batch Normalization this means that the variance state
    estimated during training and applied during evaluation may become negative. This blows up the model. In order
    to get around this issue, we modify all batch normalization layers in EfficientNet to not carry such state by
    setting track_running_stats to false.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Baseline()
        shutoff_batch_norm_tracking(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
