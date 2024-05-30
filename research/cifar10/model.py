import torch.nn.functional as F

import torch

from torch.nn import Module
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU


class ConvNet2(Module):

    def __init__(self,
                 in_channels: int,
                 h: int = 32,
                 w: int = 32,
                 hidden: int = 2048,
                 class_num: int = 10,
                 use_bn: bool = True,
                 dropout: float = .0) -> None:
        super(ConvNet2, self).__init__()

        self.conv1: Conv2d = Conv2d(in_channels, 32, 5, padding=2)
        self.conv2: Conv2d = Conv2d(32, 64, 5, padding=2)
        self.use_bn: bool = use_bn
        if use_bn:
            self.bn1: BatchNorm2d = BatchNorm2d(32)
            self.bn2: BatchNorm2d = BatchNorm2d(64)

        self.fc1: Linear = Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2: Linear = Linear(hidden, class_num)

        self.relu: ReLU = ReLU(inplace=True)
        self.maxpool: MaxPool2d = MaxPool2d(2)
        self.dropout: float = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = Flatten()(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x