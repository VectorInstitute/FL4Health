import torch
import torch.nn.functional as F
from torch import nn


class FedHeartDiseaseLargeBaseline(nn.Module):
    """
    This represents the model used by FedHeartDisease experiments leveraging a "large" model with equivalent DOF to
    the FENDA model implementation. In the current experimental setups, the FLamby Baseline model is used, which has a
    smaller number of trainable parameters. This setup corresponds to the "small" model experiments. To use the model
    here, one need only replace instances of Baseline() with FedHeartDiseaseLargeBaseline(), along with including
    the proper imports.
    """

    def __init__(self, input_dim: int = 13) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 10)
        self.fc2 = torch.nn.Linear(10, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
