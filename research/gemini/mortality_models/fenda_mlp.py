import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.fenda_base import FendaGlobalModule, FendaHeadModule, FendaJoinMode, FendaLocalModule


class FendaClassifier(FendaHeadModule):
    def __init__(self, join_mode: FendaJoinMode) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(128 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.activation = nn.ReLU()

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(input_tensor))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LocalMLP(FendaLocalModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(35, 256)
        self.fc2 = nn.Linear(256, 128)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        return self.activation(self.fc2(x))


class GlobalMLP(FendaGlobalModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(35, 256)
        self.fc2 = nn.Linear(256, 128)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        return self.activation(self.fc2(x))
