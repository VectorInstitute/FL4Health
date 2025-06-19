import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.masked_layers.masked_conv import MaskedConv2d
from fl4health.model_bases.masked_layers.masked_linear import MaskedLinear


class Masked4Cnn(nn.Module):
    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.conv1 = MaskedConv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv2 = MaskedConv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv3 = MaskedConv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.conv4 = MaskedConv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same", device=device
        )
        self.fc1 = MaskedLinear(6272, 256, device=device)
        self.fc2 = MaskedLinear(256, 256, device=device)
        self.fc3 = MaskedLinear(256, 10, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(self.conv4(x), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
