import logging
from collections.abc import Callable
from enum import Enum

import torch
from flwr.common import EvaluateRes, FitRes
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from torch import nn


TorchInputType = torch.Tensor | dict[str, torch.Tensor]
TorchTargetType = torch.Tensor | dict[str, torch.Tensor]
TorchPredType = dict[str, torch.Tensor]
TorchFeatureType = dict[str, torch.Tensor]
TorchTransformFunction = Callable[[torch.Tensor], torch.Tensor]
LayerSelectionFunction = Callable[[nn.Module, nn.Module | None], tuple[NDArrays, list[str]]]

FitFailures = list[tuple[ClientProxy, FitRes] | BaseException]
EvaluateFailures = list[tuple[ClientProxy, EvaluateRes] | BaseException]


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
