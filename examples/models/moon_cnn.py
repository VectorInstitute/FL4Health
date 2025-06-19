import torch
import torch.nn.functional as F
from torch import nn


class HeadCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 10)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.fc1(input_tensor)


class ProjectionCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(input_tensor))
        return self.fc2(x)


class BaseCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return F.relu(self.fc1(x))
