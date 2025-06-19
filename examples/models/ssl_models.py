import torch
import torch.nn.functional as F
from torch import nn


class CifarSslEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class CifarSslProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        return self.bn2(self.fc2(x))


class CifarSslPredictionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)
