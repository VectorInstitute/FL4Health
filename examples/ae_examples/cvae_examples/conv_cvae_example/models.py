import torch
import torch.nn.functional as F
from torch import nn


class ConvConditionalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_vector_size: int = 4,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2),  # Output shape: 12*12
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2),  # Output shape: 4*4
            nn.ReLU(),
        )
        self.fc = nn.Linear(16 * 4 * 4 + condition_vector_size, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(input)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Condition is a one_hot encoded vector
        x = torch.cat((x, condition), dim=-1)
        x = F.relu(self.fc(x))
        return self.fc_mu(x), self.fc_logvar(x)


class ConvConditionalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_vector_size: int = 4,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + condition_vector_size, 64)
        self.fc2 = nn.Linear(64, 16 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=2),  # Output shape: 11*11
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 5, stride=2),  # Output shape: 14*14
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=1),  # Output shape: 28*28
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_vector: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent_vector = torch.cat((latent_vector, condition), dim=-1)
        z = F.relu(self.fc1(latent_vector))
        z = F.relu(self.fc2(z))
        z = z.view(-1, 16, 4, 4)
        z = self.deconv(z)
        return self.sigmoid(z)
