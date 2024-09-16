import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU


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
        x = self.fc2(x)

        return x
