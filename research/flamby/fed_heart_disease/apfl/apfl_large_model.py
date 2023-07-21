import torch
import torch.nn as nn
import torch.nn.functional as F


class FedHeartDiseaseLargeApfl(nn.Module):
    def __init__(self, input_dim: int = 13) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 5)
        self.fc2 = torch.nn.Linear(5, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
