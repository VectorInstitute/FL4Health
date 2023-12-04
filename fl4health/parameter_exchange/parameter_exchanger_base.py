from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from flwr.common.typing import Config, NDArrays


class ParameterExchanger(ABC):
    @abstractmethod
    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        raise NotImplementedError

    @abstractmethod
    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        raise NotImplementedError
