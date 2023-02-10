from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import torch
import torch.nn as nn


class FendaJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class FendaGlobalModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    def get_layer_names(self) -> List[str]:
        # This function supplies the names of the layers to be exchanged with the central server during FL training
        # NOTE: By default, global FENDA modules will return all layer names to be exchanged. This behavior can be
        # modified by overriding this function
        return list(self.state_dict().keys())


class FendaLocalModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


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
    def __init__(
        self, local_module: FendaLocalModule, global_module: FendaGlobalModule, model_head: FendaHeadModule
    ) -> None:
        super().__init__()
        self.local_module = local_module
        self.global_module = global_module
        self.model_head = model_head

    def layers_to_exchange(self) -> List[str]:
        # NOTE: that the prepending string must match the name of the global module variable
        return [f"global_module.{layer_name}" for layer_name in self.global_module.get_layer_names()]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        local_output = self.local_module.forward(input)
        global_output = self.global_module.forward(input)
        return self.model_head.forward(local_output, global_output)
