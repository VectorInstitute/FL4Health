from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn as nn


class ParallelFeatureJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class ParallelSplitHeadModule(nn.Module, ABC):
    def __init__(self, mode: ParallelFeatureJoinMode) -> None:
        super().__init__()
        self.mode = mode

    @abstractmethod
    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> torch.Tensor:
        head_input = (
            self.parallel_output_join(first_tensor, second_tensor)
            if self.mode == ParallelFeatureJoinMode.CONCATENATE
            else torch.add(first_tensor, second_tensor)
        )
        return self.head_forward(head_input)


class ParallelSplitModel(nn.Module):
    def __init__(self, first_module: nn.Module, second_module: nn.Module, model_head: ParallelSplitHeadModule) -> None:
        super().__init__()
        self.first_module = first_module
        self.second_module = second_module
        self.model_head = model_head

    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        first_output = self.first_module.forward(input)
        second_output = self.second_module.forward(input)
        preds = {"prediction": self.model_head.forward(first_output, second_output)}
        # No features are returned in the vanilla ParallelSplitModel implementation
        return preds, {}
