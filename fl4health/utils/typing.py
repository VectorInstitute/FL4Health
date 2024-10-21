import logging
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays

TorchInputType = torch.Tensor | dict[str, torch.Tensor]
TorchTargetType = torch.Tensor | dict[str, torch.Tensor]
TorchPredType = dict[str, torch.Tensor]
TorchFeatureType = dict[str, torch.Tensor]
TorchTransformFunction = Callable[[torch.Tensor], torch.Tensor]
LayerSelectionFunction = Callable[[nn.Module, nn.Module | None], tuple[NDArrays, list[str]]]


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
