import torch
import torch.nn as nn
import torch.nn.functional as F

from fl4health.model_bases.moon_base import MoonModel


class Logistic(nn.Module):
    """Moon feature extractor module"""

    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        return x


class Head(nn.Module):
    """Moon head module"""

    def __init__(self, stack_output_dimension: int):
        super().__init__()
        self.fc1 = nn.Linear(stack_output_dimension, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dropout(input_tensor)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class FedHeartDiseaseMoonModel(MoonModel):
    def __init__(self) -> None:
        base_module = Logistic()
        head_module = Head(5)
        super().__init__(base_module, head_module)
