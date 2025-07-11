import torch
from flamby.datasets.fed_ixi import Baseline
from torch import nn

from research.flamby.utils import shutoff_batch_norm_tracking


class FedAdamUNet(nn.Module):
    """
    FedAdam implements server-side momentum in aggregating the updates from each client. For layers that carry state
    that must remain non-negative, like BatchNormalization layers (present in FedIXI U-Net), they may become negative
    due to momentum carrying updates past the origin. For Batch Normalization this means that the variance state
    estimated during training and applied during evaluation may become negative. This blows up the model. In order
    to get around this issue, we modify all batch normalization layers in the FedIXI U-Net to not carry such state by
    setting track_running_stats to false.

    **NOTE**: We set the out_channels_first_layer to 12 rather than the default of 8. This roughly doubles the size of
    the baseline model to be used (1106520 DOF). This is to allow for a fair parameter comparison with FENDA and APFL.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Baseline(out_channels_first_layer=12)
        shutoff_batch_norm_tracking(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
