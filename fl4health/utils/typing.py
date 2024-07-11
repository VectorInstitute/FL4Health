from typing import Dict, TypeVar

import torch

TorchInputType = TypeVar("TorchInputType", torch.Tensor, Dict[str, torch.Tensor])
TorchTargetType = TypeVar("TorchTargetType", torch.Tensor, Dict[str, torch.Tensor])
TorchPredType = Dict[str, torch.Tensor]
