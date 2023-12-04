import torch
import torch.nn as nn
import torch.nn.functional as F


class FedHeartDiseaseLargeApfl(nn.Module):
    """
    This represents the model used by the APFL experiments leveraging a "large" model with equivalent DOF to the FENDA
    model implementation. In the current experimental setup for APFL, the FLamby Baseline model is used, which has a
    smaller number of trainable parameters. This setup corresponds to the "small" model APFL experiments. To use the
    model here, one need only replace instances of Baseline() with FedHeartDiseaseLargeApfl(), along with including
    the proper imports.
    """

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
