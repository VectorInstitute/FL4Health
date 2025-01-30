import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, device: torch.device, dim: int = -1) -> None:
        super().__init__()
        self.cosine_similarity_function = nn.CosineSimilarity(dim=dim).to(device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Assumes that the tensors are provided "batch first" and computes the mean (over the batch) of the absolute
        value of the cosine similarity between features in x1 and x2

        Args:
            x1 (torch.Tensor): First set of tensors to compute cosine sim, shape (batch_size, n_features)
            x2 (torch.Tensor): Second set of tensors to compute cosine sim, shape (batch_size, n_features)

        Returns:
            torch.Tensor: Mean absolute value of the cosine similarity between vectors across the mutual batch size.
        """
        assert len(x1) == len(x2), "Tensors have different batch sizes"
        return torch.abs(self.cosine_similarity_function(x1, x2)).mean()
