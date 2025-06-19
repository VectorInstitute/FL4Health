import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map
from torch import nn
from torch.utils import model_zoo

from fl4health.model_bases.moon_base import MoonModel
from research.flamby.utils import shutoff_batch_norm_tracking


def from_pretrained(model_name: str, in_channels: int = 3, include_top: bool = False) -> EfficientNet:
    # There is a bug in the EfficientNet implementation if you want to strip off the top layer of the network, but
    # still load the pre-trained weights. So we do it ourselves here.
    model = EfficientNet.from_name(model_name, include_top=include_top)
    state_dict = model_zoo.load_url(url_map[model_name])
    state_dict.pop("_fc.weight")
    state_dict.pop("_fc.bias")
    model.load_state_dict(state_dict, strict=False)
    model._change_in_channels(in_channels)
    return model


class HeadClassifier(nn.Module):
    """MOON head module."""

    def __init__(self, stack_output_dimension: int):
        super().__init__()
        self.fc1 = nn.Linear(stack_output_dimension, 8)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dropout(input_tensor)
        return self.fc1(x)


class BaseEfficientNet(nn.Module):
    """
    MOON feature extractor module.

    We use the EfficientNets architecture that many participants in the ISIC competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch re-implementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    When loading the EfficientNet-B0 model, we strip off the FC layer to use the model as a feature extractor.
    There is an option to freeze a subset of the layers to reduce the number of trainable parameters. However,
    it is not used in the MOON experiments.
    """

    def __init__(self, frozen_blocks: int | None = 13, turn_off_bn_tracking: bool = False):
        super().__init__()
        # include_top ensures that we just use feature extraction in the forward pass
        self.base_model = from_pretrained("efficientnet-b0", include_top=False)
        if frozen_blocks:
            self.freeze_layers(frozen_blocks)
        if turn_off_bn_tracking:
            shutoff_batch_norm_tracking(self.base_model)

    def freeze_layers(self, frozen_blocks: int) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate through the blocks freezing the specified number up to 15 (all of them)

        # Freeze the first two layers
        self.base_model._modules["_conv_stem"].requires_grad_(False)
        self.base_model._modules["_bn0"].requires_grad_(False)
        # Now we iterate through the block modules and freeze a certain number of them.
        frozen_blocks = min(frozen_blocks, 15)
        for block_index in range(frozen_blocks):
            self.base_model._modules["_blocks"][block_index].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x.flatten(start_dim=1)


class FedIsic2019MoonModel(MoonModel):
    def __init__(self, frozen_blocks: int | None = None, turn_off_bn_tracking: bool = False) -> None:
        base_module = BaseEfficientNet(frozen_blocks, turn_off_bn_tracking=turn_off_bn_tracking)
        head_module = HeadClassifier(1280)
        super().__init__(base_module, head_module)
