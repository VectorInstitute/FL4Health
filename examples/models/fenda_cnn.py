from typing import Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl4health.model_bases.fenda_base import FendaClassifierModule, FendaGlobalModule, FendaJoinMode, FendaLocalModule


class FendaClassifier(FendaClassifierModule):
    def __init__(self, join_mode: FendaJoinMode) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(120 * 2, 84)
        self.fc2 = nn.Linear(84, 10)

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def classifier_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(input_tensor))
        x = self.fc2(x)
        return x


class LocalCnn(FendaLocalModule):
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
        x = F.relu(self.fc1(x))
        return x


class GlobalCnn(FendaGlobalModule):
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
        x = F.relu(self.fc1(x))
        return x

    def get_layer_names(self) -> Set[str]:
        return Set(self.state_dict().keys())
