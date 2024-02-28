from collections import OrderedDict
from logging import DEBUG
from typing import List, Optional

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    discrete_gaussian_noise_vector,
    generate_sign_diagonal_matrix,
    generate_walsh_hadamard_matrix,
    pad_zeros,
    randomized_rounding,
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.server.secure_aggregation_utils import get_model_norm, vectorize_model


class SecureAggregationExchanger(ParameterExchanger):
    # def push_parameters(
    #     self,
    #     *,
    #     model: nn.Module,
    #     mask: List[int],
    #     scalar: int,
    #     dp_setting: Config,
    #     dtype: torch.Tensor = torch.float64,
    #     initial_model: Optional[nn.Module] = None,
    #     config: Optional[Config] = None
    # ) -> NDArrays:
    #     # This is more than an exchanger: also post processing for secure aggergation for the client
    #     # Sending all of parameters ordered by state_dict keys
    #     # NOTE: Order matters because it is relied upon by pull_parameters below

    #     # used to observe mask cancellation
    #     debug_mode = False

    #     dim = sum(param.numel() for param in model.parameters() if param.requires_grad)

    #     assert dim == len(mask)  # mask len and model dim must match

    #     if dp_setting["dp_mechanism"] == PrivacyMechanismIndex.DiscreteGaussian.value:

    #         # compute noise vector
    #         variance = (dp_setting["noise_scale"] / dp_setting["granularity"]) ** 2
    #         noise_vector = discrete_gaussian_noise_vector(d=dim, variance=variance)

    #         # clipping
    #         c = dp_setting["clipping_threshold"]
    #         assert c > 0  # clipping threshold needs to be positive
    #         # the scalar is the weighting by training sample size
    #         clipping_scalar = torch.min(1, c / (scalar * get_model_norm(model))) / dp_setting["granularity"]
    #         clipped_model_vector = (clipping_scalar * scalar) * vectorize_model(model)
    #         padded_model_vector = pad_zeros(clipped_model_vector)
    #         del clipped_model_vector

    #         # used for matrix transform
    #         diagonal_matrix = generate_sign_diagonal_matrix(dim)
    #         welsh_hadamard_matrix = generate_walsh_hadamard_matrix(exponent=torch.ceil(torch.log2(dim)))

    #         # TODO make diagonal multiplication more efficient
    #         flattened_vector = torch.matmul(
    #             input=welsh_hadamard_matrix, other=torch.matmul(input=diagonal_matrix, other=padded_model_vector)
    #         )
    #         del diagonal_matrix, welsh_hadamard_matrix, clipped_model_vector

    #         rounded_vector = randomized_rounding(
    #             flattened_vector, dp_setting["clipping_threshold"], dp_setting["granularity"], dim, dp_setting["bias"]
    #         )
    #         del flattened_vector
    #         noised_and_masked_vector = rounded_vector + noise_vector + torch.tensor(mask, dtype=dtype)
    #         del rounded_vector

    #     # pass in online_clients kwarg for drop out case
    #     # modify parms  << quantization for privacy mechanism + modular arithmetic>>
    #     model.to(dtype)
    #     masked_model_layers = []
    #     i = 0
    #     params_dict = model.state_dict()
    #     for name, params in params_dict.items():
    #         j = i + params.numel()
    #         # layer_mask_tensor = torch.tensor(mask[i:j], dtype=dtype).reshape(params.size())
    #         # noise_tensor = torch.tensor(noise_vector[i:j], dtype=dtype).reshape(params.size())
    #         # TEST ONLY #
    #         # if debug_mode:
    #         #     masked_model_layers.append(layer_mask_tensor.cpu().numpy())
    #         #     i = j
    #         #     continue

    #         # NOTE scaling must happen before masking
    #         # params = noise_vector[i:j].reshape(params.size())
    #         # scaled_params = scalar * params

    #         # masking & DP-processing
    #         # masked_layer = scaled_params + layer_mask_tensor + noise_tensor
    #         masked_layer = noised_and_masked_vector[i:j].reshape(params.size())
    #         masked_layer %= dp_setting["arithmetic_modulus"]
    #         masked_model_layers.append(masked_layer.cpu().numpy())
    #         i = j
    #     # log(DEBUG, 'client updates====')
    #     # log(DEBUG, masked_model_layers[0])
    #     return masked_model_layers

    # def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Optional[Config] = None) -> None:
    #     # Assumes all model parameters are contained in parameters
    #     # The state_dict is reconstituted because parameters is simply a list of bytes
    #     # log(DEBUG, parameters)
    #     params_dict = zip(model.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     model.load_state_dict(state_dict, strict=True)

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
