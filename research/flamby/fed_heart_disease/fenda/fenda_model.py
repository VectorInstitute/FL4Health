import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode, ParallelSplitHeadModule


class FendaClassifier(ParallelSplitHeadModule):
    def __init__(self, join_mode: ParallelFeatureJoinMode, stack_output_dimension: int) -> None:
        super().__init__(join_mode)
        # Two layer DNN as a classifier head
        self.fc1 = nn.Linear(stack_output_dimension * 2, 1)
        self.dropout = nn.Dropout(0.3)

    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        local_tensor = local_tensor.flatten(start_dim=1)
        global_tensor = global_tensor.flatten(start_dim=1)
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dropout(input_tensor)
        x = self.fc1(x)
        return torch.sigmoid(x)


class LocalLogistic(nn.Module):
    """Local FENDA module."""

    def __init__(self, input_dim: int = 13):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return F.relu(x)


class GlobalLogistic(nn.Module):
    """Global FENDA module."""

    def __init__(self, input_dim: int = 13) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return F.relu(x)


class FedHeartDiseaseFendaModel(FendaModel):
    def __init__(self) -> None:
        local_module = LocalLogistic()
        global_module = GlobalLogistic()
        model_head = FendaClassifier(ParallelFeatureJoinMode.CONCATENATE, 5)
        super().__init__(local_module, global_module, model_head)
