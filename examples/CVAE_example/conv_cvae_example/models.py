from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvConditionalEncoder(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2),  # 12*12
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, stride=2),  # 4*4
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(16 * 4 * 4 + num_conditions, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(input)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Condition is a one_hot encoded vector
        x = torch.cat((x, condition), dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)


class ConvConditionalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_conditions: int,
    ) -> None:
        super().__init__()

        self.fc4 = nn.Linear(latent_dim + num_conditions, 64)
        self.fc5 = nn.Linear(64, 16 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 5, stride=2),  # 11*11
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 5, stride=2),  # 14*14
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4, stride=1),  # 28*28
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        # Decoder gets the concatnated tensor of latent sample and condition
        z = F.relu(self.fc4(latent_vector))
        z = F.relu(self.fc5(z))
        z = z.view(-1, 16, 4, 4)
        z = self.deconv(z)

        return self.sigmoid(z)
