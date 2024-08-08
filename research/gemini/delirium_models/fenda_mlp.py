import torch
import torch.nn as nn

from fl4health.model_bases.fenda_base import FendaGlobalModule, FendaHeadModule, FendaJoinMode, FendaLocalModule


class FendaClassifier_d(FendaHeadModule):
    def __init__(self, join_mode: FendaJoinMode, size: int = 136) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(size, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        local_tensor = local_tensor.flatten(start_dim=1)
        global_tensor = global_tensor.flatten(start_dim=1)
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dropout(input_tensor)
        x = self.fc1(x)
        return x


class LocalMLP_d(FendaLocalModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8093, 256 * 2)
        self.fc2 = nn.Linear(256 * 2, 8)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        return x


class GlobalMLP_d(FendaGlobalModule):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8093, 256 * 2)
        self.fc2 = nn.Linear(256 * 2, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        return x
