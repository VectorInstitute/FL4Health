import torch
import torch.nn.functional as F
from torch import nn


class MnistConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_dim: int,
        condition_vector_size: int = 2,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size + condition_vector_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.cat((input, condition), dim=-1)
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return self.fc_mu(x), self.fc_logvar(x)


class MnistConditionalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_size: int,
        condition_vector_size: int = 2,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + condition_vector_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, latent_vector: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent_vector = torch.cat((latent_vector, condition), dim=-1)
        latent_vector = F.relu(self.fc1(latent_vector))
        latent_vector = F.relu(self.fc2(latent_vector))
        return F.sigmoid(self.fc3(latent_vector))
