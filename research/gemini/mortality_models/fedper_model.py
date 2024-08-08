import torch
import torch.nn as nn

from fl4health.model_bases.fedper_base import FedPerModel


class FedPerGloalFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 2)
        self.fc2 = nn.Linear(256 * 2, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x


class FedPerLocalPredictionHead(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc5(x)
        return x


class GeminiFedPerModel(FedPerModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        base_module = FedPerGloalFeatureExtractor(input_dim)
        head_module = FedPerLocalPredictionHead(output_dim)
        super().__init__(base_module, head_module)
