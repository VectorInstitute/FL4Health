from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger


class FullParameterExchanger(ParameterExchanger):
    def push_parameters(
        self, model: nn.Module, initial_model: Optional[nn.Module] = None, config: Optional[Config] = None
    ) -> NDArrays:
        # Sending all of parameters ordered by state_dict keys
        # NOTE: Order matters, because it is relied upon by pull_parameters below
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        # Assumes all model parameters are contained in parameters
        # The state_dict is reconstituted because parameters is simply a list of bytes
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
