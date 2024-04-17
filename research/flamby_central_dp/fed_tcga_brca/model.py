import torch.nn as nn 
import torch
import torch.nn.functional as F

class Fed_TCGA_BRCA_LargeBaseline(nn.Module):

    def __init__(self, input_dim: int = 39) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 50)
        self.fc2 = torch.nn.Linear(50, 25)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)