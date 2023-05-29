from logging import INFO
from pathlib import Path
from typing import Optional, Tuple

import torch
from flwr.common import Config, NDArrays
from flwr.common.logger import log
from numpy import linalg

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithClippingBit


class NumpyClippingClient(NumpyFlClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        super().__init__(data_path, device)
        self.parameter_exchanger: ParameterExchangerWithClippingBit
        self.clipping_bound: Optional[float] = None
        self.adaptive_clipping: Optional[bool] = None

    def calculate_parameters_norm(self, parameters: NDArrays) -> float:
        layer_inner_products = [pow(linalg.norm(layer_weights), 2) for layer_weights in parameters]
        # network froebenius norm
        return pow(sum(layer_inner_products), 0.5)

    def clip_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, float]:
        assert self.clipping_bound is not None
        assert self.adaptive_clipping is not None
        # performs flat clipping (i.e. parameters * min(1, C/||parameters||_2))
        network_frobenius_norm = self.calculate_parameters_norm(parameters)
        log(INFO, f"Update norm: {network_frobenius_norm}, Clipping Bound: {self.clipping_bound}")
        if network_frobenius_norm <= self.clipping_bound:
            # if we're not adaptively clipping then don't send true clipping bit info as this would potentially leak
            # information
            clipping_bit = 1.0 if self.adaptive_clipping else 0.0
            return parameters, clipping_bit
        clip_scalar = min(1.0, self.clipping_bound / network_frobenius_norm)
        # parameters and clipping bit
        return [layer_weights * clip_scalar for layer_weights in parameters], 0.0

    def compute_weight_update_and_clip(self, parameters: NDArrays) -> Tuple[NDArrays, float]:
        assert self.initial_weights is not None
        weight_update: NDArrays = [
            new_layer_weights - old_layer_weights
            for old_layer_weights, new_layer_weights in zip(self.initial_weights, parameters)
        ]
        # return clipped parameters and clipping bit
        return self.clip_parameters(weight_update)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        This function performs clipping through compute_weight_update_and_clip and stores the clipping bit
        as the last entry in the NDArrays
        """
        assert self.model is not None and self.parameter_exchanger is not None
        model_weights = self.parameter_exchanger.push_parameters(self.model, config)
        clipped_weight_update, clipping_bit = self.compute_weight_update_and_clip(model_weights)
        return self.parameter_exchanger.pack_parameters(clipped_weight_update, clipping_bit)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        This function assumes that the parameters being passed contain model parameters followed by the last entry
        of the list being the new clipping bound.
        """
        assert self.model is not None and self.parameter_exchanger is not None
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        server_model_parameters, clipping_bound = self.parameter_exchanger.unpack_parameters(parameters)
        # Store the starting parameters without clipping bound before client optimization steps
        self.initial_weights = server_model_parameters
        self.clipping_bound = clipping_bound
        # Inject the server model parameters into the client model
        self.parameter_exchanger.pull_parameters(server_model_parameters, self.model, config)
