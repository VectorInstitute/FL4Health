import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map
from torch.utils import model_zoo

from fl4health.model_bases.fenda_base import (
    FendaGlobalModule,
    FendaHeadModule,
    FendaJoinMode,
    FendaLocalModule,
    FendaModel,
)


def from_pretrained(model_name: str, in_channels: int = 3, include_top: bool = False) -> EfficientNet:
    model = EfficientNet.from_name(model_name, include_top=include_top)
    state_dict = model_zoo.load_url(url_map[model_name])
    state_dict.pop("_fc.weight")
    state_dict.pop("_fc.bias")
    model.load_state_dict(state_dict, strict=False)
    model._change_in_channels(in_channels)
    return model


class FendaClassifier(FendaHeadModule):
    def __init__(self, join_mode: FendaJoinMode, stack_output_dimension: int) -> None:
        super().__init__(join_mode)
        # Two layer DNN as a classifier head
        self.fc1 = nn.Linear(stack_output_dimension * 2, 64)
        self.fc2 = nn.Linear(64, 8)
        self.dropout = nn.Dropout(0.2)

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        local_tensor = local_tensor.flatten(start_dim=1)
        global_tensor = global_tensor.flatten(start_dim=1)
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dropout(input_tensor)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LocalEfficientNet(FendaLocalModule):
    """Local FENDA module
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    When loading the EfficientNet-B0 model, we strip off the FC layer to use the model as a feature extractor.
    We freeze a subset of the layers in order to make sure that FENDA is not training twice as many parameters as the
    other approaches.
    """

    def __init__(self, frozen_blocks: int = 13):
        super().__init__()
        # include_top ensures that we just use feature extraction in the forward pass
        self.base_model = from_pretrained("efficientnet-b0", include_top=False)
        self.freeze_layers(frozen_blocks)

    def freeze_layers(self, frozen_blocks: int) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate throught the blocks freezing the specified number up to 15 (all of them)

        # Freeze the first two layers
        self.base_model._modules["_conv_stem"].requires_grad_(False)
        self.base_model._modules["_bn0"].requires_grad_(False)
        # Now we iterate through the block modules and freeze a certain number of them.
        frozen_blocks = min(frozen_blocks, 15)
        for block_index in range(frozen_blocks):
            self.base_model._modules["_blocks"][block_index].requires_grad_(False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x


class GlobalEfficientNet(FendaGlobalModule):
    """Global FENDA module
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    When loading the EfficientNet-B0 model, we strip off the FC layer to use the model as a feature extractor.
    We freeze a subset of the layers in order to make sure that FENDA is not training twice as many parameters as the
    other approaches.
    """

    def __init__(self, frozen_blocks: int = 14):
        super().__init__()
        # include_top ensures that we just use feature extraction in the forward pass
        self.base_model = from_pretrained("efficientnet-b0", include_top=False)
        self.freeze_layers(frozen_blocks)

    def freeze_layers(self, frozen_blocks: int) -> None:
        # We freeze the bottom layers of the network. We always freeze the _conv_stem module, the _bn0 module and then
        # we iterate throught the blocks freezing the specified number up to 15 (all of them)

        # Freeze the first two layers
        self.base_model._modules["_conv_stem"].requires_grad_(False)
        self.base_model._modules["_bn0"].requires_grad_(False)
        # Now we iterate through the block modules and freeze a certain number of them.
        frozen_blocks = min(frozen_blocks, 15)
        for block_index in range(frozen_blocks):
            self.base_model._modules["_blocks"][block_index].requires_grad_(False)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return x


class FedIsic2019FendaModel(FendaModel):
    def __init__(self) -> None:
        local_module = LocalEfficientNet()
        global_module = GlobalEfficientNet()
        model_head = FendaClassifier(FendaJoinMode.CONCATENATE, 1280)
        super().__init__(local_module, global_module, model_head)
