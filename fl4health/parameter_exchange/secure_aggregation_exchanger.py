from collections import OrderedDict
from logging import DEBUG
from typing import List, Optional

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger


class SecureAggregationExchanger(ParameterExchanger):
    def push_parameters(
        self,
        *,
        model: nn.Module,
        mask: List[int],
        dtype: torch.Tensor = torch.float64,
        initial_model: Optional[nn.Module] = None,
        config: Optional[Config] = None
    ) -> NDArrays:
        # Sending all of parameters ordered by state_dict keys
        # NOTE: Order matters because it is relied upon by pull_parameters below

        # used to observe mask cancellation
        debug_mode = False

        dim = sum(param.numel() for param in model.parameters() if param.requires_grad)

        assert dim == len(mask)  # mask len and model dim must match

        # pass in online_clients kwarg for drop out case
        # modify parms  << quantization for privacy mechanism + modular arithmetic>>
        model.to(dtype)
        masked_model_layers = []
        i = 0
        params_dict = model.state_dict()
        for name, params in params_dict.items():
            j = i + params.numel()
            layer_mask_tensor = torch.tensor(mask[i:j], dtype=dtype).reshape(params.size())
            # TEST ONLY #
            if debug_mode:
                masked_model_layers.append(layer_mask_tensor.cpu().numpy())
                i = j
                continue

            masked_layer = params + layer_mask_tensor  # + add appropriate DP mechanism
            masked_model_layers.append(masked_layer.cpu().numpy())
            i = j
        # log(DEBUG, 'client updates====')
        # log(DEBUG, masked_model_layers[0])
        return masked_model_layers

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
        # Assumes all model parameters are contained in parameters
        # The state_dict is reconstituted because parameters is simply a list of bytes
        # log(DEBUG, parameters)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
