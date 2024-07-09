from typing import Dict, List, Tuple, TypeVar, Union

import torch

TorchInputType = TypeVar("TorchInputType", torch.Tensor, Dict[str, torch.Tensor])
TorchTargetType = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
TorchPredType = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
