import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU

from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode, ParallelSplitHeadModule
from fl4health.model_bases.sequential_split_models import SequentiallySplitModel


class ConvNet(Module):
    def __init__(
        self,
        in_channels: int,
        h: int = 32,
        w: int = 32,
        hidden: int = 2048,
        class_num: int = 10,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.flatten = Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = self.flatten(x)
        x = self.dropout_layer(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(x)
        return self.fc2(x)


class ConvNetFeatureExtractor(Module):
    def __init__(
        self,
        in_channels: int,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        self.conv1 = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = BatchNorm2d(32)
            self.bn2 = BatchNorm2d(64)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.flatten = Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        return self.flatten(x)


class ConvNetClassifier(Module):
    def __init__(
        self,
        h: int = 32,
        w: int = 32,
        hidden: int = 2048,
        class_num: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(x)
        return self.fc2(x)


class ConvNetFendaClassifier(ParallelSplitHeadModule):
    def __init__(
        self,
        join_mode: ParallelFeatureJoinMode,
        h: int = 32,
        w: int = 32,
        hidden: int = 2048,
        class_num: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(join_mode)

        # Times 2 because we'll be concatenating the inputs of two feature extractors
        self.fc1 = Linear((h // 2 // 2) * (w // 2 // 2) * 64 * 2, hidden)
        self.fc2 = Linear(hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(p=dropout)

    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first", we concatenate along the channel dimension
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout_layer(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return self.fc2(x)


class ConvNetFendaModel(FendaModel):
    def __init__(
        self,
        in_channels: int,
        h: int = 32,
        w: int = 32,
        hidden: int = 2048,
        class_num: int = 10,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        # FedIXI out_channels_first_layer = 8 is the Baseline model default. So we use it here. The monte carlo dropout
        # is also set to 0 by default for FedIXI
        local_module = ConvNetFeatureExtractor(in_channels, use_bn)
        global_module = ConvNetFeatureExtractor(in_channels, use_bn)
        model_head = ConvNetFendaClassifier(
            ParallelFeatureJoinMode.CONCATENATE, h=h, w=w, hidden=hidden, class_num=class_num, dropout=dropout
        )
        super().__init__(local_module=local_module, global_module=global_module, model_head=model_head)


class ConvNetFendaDittoGlobalModel(SequentiallySplitModel):
    def __init__(
        self,
        in_channels: int,
        h: int = 32,
        w: int = 32,
        hidden: int = 2048,
        class_num: int = 10,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        base_module = ConvNetFeatureExtractor(in_channels, use_bn)
        head_module = ConvNetClassifier(h=h, w=w, hidden=hidden, class_num=class_num, dropout=dropout)
        super().__init__(base_module, head_module, flatten_features=False)
