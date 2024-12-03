from abc import ABC, abstractmethod

import torch.nn as nn


class PartialLayerExchangeModel(nn.Module, ABC):
    @abstractmethod
    def layers_to_exchange(self) -> list[str]:
        raise NotImplementedError
