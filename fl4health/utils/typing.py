import logging
from collections.abc import Callable
from enum import Enum
from typing import List, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from flwr.common import EvaluateRes, FitRes
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger

TorchInputType = torch.Tensor | dict[str, torch.Tensor]
TorchTargetType = torch.Tensor | dict[str, torch.Tensor]
TorchPredType = dict[str, torch.Tensor]
TorchFeatureType = dict[str, torch.Tensor]
TorchTransformFunction = Callable[[torch.Tensor], torch.Tensor]
LayerSelectionFunction = Callable[[nn.Module, nn.Module | None], tuple[NDArrays, list[str]]]

FitFailures = List[Union[Tuple[ClientProxy, FitRes], BaseException]]
EvaluateFailures = List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
