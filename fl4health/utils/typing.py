import logging
from enum import Enum
from typing import Callable, Dict, Tuple, Union

import torch

TorchInputType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TorchTargetType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TorchPredType = Dict[str, torch.Tensor]
TorchFeatureType = Dict[str, torch.Tensor]
TorchTransformType = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
