import torch
from flamby.datasets.fed_ixi import Baseline
from torch import nn

from research.flamby.utils import shutoff_batch_norm_tracking


class ApflUNet(nn.Module):
    """
    APFL module to serve as both the local and global models APFL unifies the logits through a convex combination of
    the local and global model versions, so we maintain the original structure of efficient net and simply interpolate
    the outputs.
    """

    def __init__(self, turn_off_bn_tracking: bool = False):
        super().__init__()
        self.base_model = Baseline()
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
