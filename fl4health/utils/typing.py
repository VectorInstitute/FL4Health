import logging
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays

TorchInputType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TorchTargetType = Union[torch.Tensor, Dict[str, torch.Tensor]]
TorchPredType = Dict[str, torch.Tensor]
TorchFeatureType = Dict[str, torch.Tensor]
TorchTransformType = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
LayerSelectionFunction = Callable[[nn.Module, Optional[nn.Module]], Tuple[NDArrays, List[str]]]


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
