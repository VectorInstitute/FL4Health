import torch
from torch import nn

from fl4health.model_bases.fenda_base import FendaHeadModule, FendaJoinMode, FendaModel


class PerFclClassifier(FendaHeadModule):
    def __init__(self, join_mode: FendaJoinMode, output_dim: int) -> None:
        super().__init__(join_mode)
        self.fc6 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        local_tensor = local_tensor.flatten(start_dim=1)
        global_tensor = global_tensor.flatten(start_dim=1)
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc6(x)


class LocalLogistic(nn.Module):
    """Local FENDA module."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        return self.activation(self.fc4(x))


class GlobalLogistic(nn.Module):
    """Global FENDA module."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        return self.activation(self.fc4(x))


class DeliriumPerFclModel(FendaModel):
    def __init__(self, input_dim, output_dim) -> None:
        local_module = LocalLogistic(input_dim)
        global_module = GlobalLogistic(input_dim)
        model_head = PerFclClassifier(FendaJoinMode.CONCATENATE, output_dim)
        super().__init__(local_module, global_module, model_head)
