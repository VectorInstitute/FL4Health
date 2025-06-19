import torch
import torch.nn.functional as F
from torch import nn


class SequentialLocalPredictionHeadMnist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(input_tensor))
        return self.fc2(x)


class SequentialGlobalFeatureExtractorMnist(nn.Module):
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


class SequentialLocalPredictionHeadCifar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(input_tensor))
        return self.fc2(x)


class SequentialGlobalFeatureExtractorCifar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return F.relu(self.fc1(x))
