import torch
from torch import nn

from fl4health.model_bases.moon_base import MoonModel


class BaseModule(nn.Module):
    """Moon feature extractor module."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(8093, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 256 * 2)
        self.fc3 = nn.Linear(256 * 2, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
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
        x = self.dropout(x)
        return self.activation(self.fc5(x))


class HeadClassifier(nn.Module):
    """Moon head module."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.fc6 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc6(x)


class DeliriumMoonModel(MoonModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        base_module = BaseModule(input_dim)
        head_module = HeadClassifier(output_dim)
        super().__init__(base_module, head_module)
