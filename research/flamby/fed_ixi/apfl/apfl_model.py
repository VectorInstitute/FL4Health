import torch
import torch.nn as nn
from flamby.datasets.fed_ixi import Baseline

from research.flamby.utils import shutoff_batch_norm_tracking


class APFLUNet(nn.Module):
    """
    APFL module to serve as both the local and global models APFL unifies the logits through a convex combination of
    the local and global model versions, so we maintain the original structure of efficient net and simply interpolate
    the outputs.

    We freeze a subset of the layers in order to make sure that APFL is not training twice as many parameters as the
    other approaches.
    """

    def __init__(self, frozen_blocks: int = 13, turn_off_bn_tracking: bool = False):
        super().__init__()
        self.base_model = Baseline()
        # Freeze layers to reduce trainable parameters.
        self.freeze_layers(frozen_blocks)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def freeze_layers(self, frozen_blocks: int) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate throught the blocks freezing the specified number up to 15 (all of them)

        # Freeze the first two layers
        self.base_model._modules["base_model"]._modules["_conv_stem"].requires_grad_(False)
        self.base_model._modules["base_model"]._modules["_bn0"].requires_grad_(False)
        # Now we iterate through the block modules and freeze a certain number of them.
        frozen_blocks = min(frozen_blocks, 15)
        for block_index in range(frozen_blocks):
            self.base_model._modules["base_model"]._modules["_blocks"][block_index].requires_grad_(False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x
