import torch
from fl4health.model_bases.fedper_base import FedPerModel
from torch import nn


class FedPerGlobalFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 4)
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


class FedPerGlobalFeatureExtractorNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 2)
        self.fc2 = nn.Linear(256 * 2, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
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
        x = self.activation(self.fc4(x))
        return self.dropout(x)


class FedPerLocalPredictionHead(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc6(x)


class DeliriumFedPerModel(FedPerModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        base_module = FedPerGlobalFeatureExtractorNet(input_dim)
        head_module = FedPerLocalPredictionHead(output_dim)
        super().__init__(base_module, head_module)
