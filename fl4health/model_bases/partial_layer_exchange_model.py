from abc import ABC, abstractmethod

from torch import nn


class PartialLayerExchangeModel(nn.Module, ABC):
    @abstractmethod
    def layers_to_exchange(self) -> list[str]:
        raise NotImplementedError
