import torch
from torch import nn
from torch.nn import Linear, Module


class FullyConnectedNet(Module):
    def __init__(
        self,
        in_channels: int = 60,
        hidden: int = 20,
        class_num: int = 10,
    ) -> None:
        super().__init__()

        self.linear_1 = Linear(in_channels, hidden, bias=True)
        self.linear_2 = Linear(hidden, class_num, bias=True)
        self.softmax_2 = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        return self.softmax_2(self.linear_2(x))
