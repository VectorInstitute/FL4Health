from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn


class FendaJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class FendaHeadModule(nn.Module, ABC):
    def __init__(self, mode: FendaJoinMode) -> None:
        super().__init__()
        self.mode = mode

    @abstractmethod
    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        head_input = (
            self.local_global_concat(local_tensor, global_tensor)
            if self.mode == FendaJoinMode.CONCATENATE
            else torch.add(local_tensor, global_tensor)
        )
        return self.head_forward(head_input)


class FendaModel(nn.Module):
    def __init__(self, local_module: nn.Module, global_module: nn.Module, model_head: FendaHeadModule) -> None:
        super().__init__()
        self.local_module = local_module
        self.global_module = global_module
        self.model_head = model_head

    def layers_to_exchange(self) -> List[str]:
        return [layer_name for layer_name in self.state_dict().keys() if layer_name.startswith("global_module.")]

    def forward(self, input: torch.Tensor, pre_train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pre_train:
            self.local_module.eval()
        else:
            self.local_module.train()
        local_output = self.local_module.forward(input)
        if pre_train:
            local_output = 0.0 * local_output
        global_output = self.global_module.forward(input)
        return self.model_head.forward(local_output, global_output), local_output, global_output
