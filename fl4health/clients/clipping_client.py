from logging import INFO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import Config, NDArrays
from flwr.common.logger import log
from numpy import linalg

from fl4health.clients.numpy_fl_client import NumpyFlClient


class NumpyClippingClient(NumpyFlClient):
    def __init__(self, data_path: Path, device: torch.device) -> None:
        super().__init__(data_path, device)
        self.clipping_bound: Optional[float] = None
        self.initial_weights: Optional[NDArrays] = None
        self.adaptive_clipping: Optional[bool] = None

    def clip_and_pack_parameters(self, parameters: NDArrays) -> NDArrays:
        clipped_weight_update, clipping_bit = self.compute_weight_update_and_clip(parameters)
        return clipped_weight_update + [np.array([clipping_bit])]

    def unpack_parameters_with_clipping_bound(self, packed_parameters: NDArrays) -> NDArrays:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        server_model_parameters = packed_parameters[:-1]
        # Store the starting parameters without clipping bound before client optimization steps
        self.initial_weights = server_model_parameters
        clipping_bound = packed_parameters[-1]
        self.clipping_bound = float(clipping_bound)
        return server_model_parameters

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
        model_weights = self.parameter_exchanger.push_parameters(self.model, config)
        return self.clip_and_pack_parameters(model_weights)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        This function assumes that the parameters being passed contain model parameters followed by the last entry
        of the list being the new clipping bound.
        """
        server_model_parameters = self.unpack_parameters_with_clipping_bound(parameters)
        self.parameter_exchanger.pull_parameters(server_model_parameters, self.model, config)

    def _first_round_fit(self, parameters: NDArrays, configs: Config) -> Tuple[List, int, Dict]:
        # To be called on first round when weighted_averaging and adaptive_clipping are true

        log(
            INFO,
            """Solely fetching client sample counts. Parameters not updated.""",
        )

        return (
            parameters,
            self.num_examples["train_set"],
            {},
        )

    def __getattribute__(self, name: str) -> Any:
        # If attribute is not fit, regular __getattribute__ behaviour
        if name != "fit":
            return object.__getattribute__(self, name)

        # Wrapper for the fit function only
        def wrapper(parameters: NDArrays, config: Config) -> Callable:
            # If client is not initialized, setup client
            if name == "fit" and self.initialized is False:
                self.setup_client(config)

                # Check if we should train
                training = config.get("training", True)

                # If true then call _first_round_fit method
                if training is False:
                    return object.__getattribute__(self, "_first_round_fit")(parameters, config)

            # Else call fit method
            return object.__getattribute__(self, "fit")(parameters, config)

        return wrapper
