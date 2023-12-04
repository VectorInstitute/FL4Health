import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.sigmoid(self.linear(x)).reshape(-1)
        return outputs
