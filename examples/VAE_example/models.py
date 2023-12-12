from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistVariationalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.fc_mu(x), self.fc_logvar(x)


class MnistVariationalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, output_size)

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        latent_vector = F.relu(self.fc4(latent_vector))
        latent_vector = F.relu(self.fc5(latent_vector))
        return F.sigmoid(self.fc6(latent_vector))
