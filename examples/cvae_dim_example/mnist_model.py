import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x