import torch
from flamby.datasets.fed_isic2019 import Baseline
from torch import nn

from research.flamby.utils import shutoff_batch_norm_tracking


class ApflEfficientNet(nn.Module):
    """
    APFL module to serve as both the local and global models.

    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    APFL unifies the logits through a convex combination of the local and global model versions, so we maintain the
    original structure of efficient net and simply interpolate the outputs.
    We freeze a subset of the layers in order to make sure that APFL is not training twice as many parameters as the
    other approaches.
    """

    def __init__(self, frozen_blocks: int | None = 13, turn_off_bn_tracking: bool = False):
        super().__init__()
        self.base_model = Baseline()
        # Freeze layers to reduce trainable parameters.
        if frozen_blocks:
            self.freeze_layers(frozen_blocks)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def freeze_layers(self, frozen_blocks: int) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate through the blocks freezing the specified number up to 15 (all of them)

        # Freeze the first two layers
        self.base_model._modules["base_model"]._modules["_conv_stem"].requires_grad_(False)
        self.base_model._modules["base_model"]._modules["_bn0"].requires_grad_(False)
        # Now we iterate through the block modules and freeze a certain number of them.
        frozen_blocks = min(frozen_blocks, 15)
        for block_index in range(frozen_blocks):
            self.base_model._modules["base_model"]._modules["_blocks"][block_index].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
