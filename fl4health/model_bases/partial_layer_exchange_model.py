from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class PartialLayerExchangeModel(nn.Module, ABC):
    @abstractmethod
    def layers_to_exchange(self) -> List[str]:
        raise NotImplementedError
