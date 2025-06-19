import torch
import torch.nn.functional as F
from torch import nn


class ConfigurableMnistNet(nn.Module):
    def __init__(self, out_channel_mult: int) -> None:
        super().__init__()
        self.out_channel_mult = out_channel_mult
        self.conv1 = nn.Conv2d(1, 8 * out_channel_mult, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8 * out_channel_mult, 16 * out_channel_mult, 5)
        self.fc1 = nn.Linear(16 * 4 * 4 * out_channel_mult, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4 * self.out_channel_mult)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
