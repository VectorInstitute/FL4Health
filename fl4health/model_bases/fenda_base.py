from abc import ABC, abstractmethod
from enum import Enum
from typing import Set

import torch
import torch.nn as nn


class FendaJoinMode(Enum):
    CONCATENATE = "CONCATENATE"
    SUM = "SUM"


class FendaGlobalModule(nn.Module, ABC):
    @abstractmethod
    def get_layer_names(self) -> Set[str]:
        raise NotImplementedError


class FendaLocalModule(nn.Module):
    pass


class FendaClassifierModule(nn.Module, ABC):
    def __init__(self, mode: FendaJoinMode) -> None:
        self.mode = mode

    @abstractmethod
    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def classifier_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        classifier_input = (
            self.local_global_concat(local_tensor, global_tensor)
            if self.mode == FendaJoinMode.CONCATENATE
            else torch.add(local_tensor, global_tensor)
        )
        return self.classifier_forward(classifier_input)


class FendaModel(nn.Module):
    def __init__(
        self, local_module: FendaLocalModule, global_module: FendaGlobalModule, classifier: FendaClassifierModule
    ) -> None:
        self.local_module = local_module
        self.global_module = global_module
        self.classifier = classifier

    def layers_to_exchange(self) -> Set[str]:
        return self.global_module.get_layer_names()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        local_output = self.local_module.forward(input)
        global_output = self.global_module.forward(input)
        return self.classifier.forward(local_output, global_output)
