from logging import INFO
from typing import Optional, Tuple

from flwr.client import NumPyClient
from flwr.common import Config, NDArrays
from flwr.common.logger import log
from numpy import linalg


class NumpyClippingClient(NumPyClient):
    def __init__(self, adaptive_clipping: bool = False) -> None:
        self.clipping_bound: Optional[float] = None
        self.current_weights: Optional[NDArrays] = None
        self.adaptive_clipping = adaptive_clipping

    def calculate_parameters_norm(self, parameters: NDArrays) -> float:
        layer_inner_products = [pow(linalg.norm(layer_weights), 2) for layer_weights in parameters]
        # network froebenius norm
        return pow(sum(layer_inner_products), 0.5)

    def clip_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, float]:
        assert self.clipping_bound is not None
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
        assert self.current_weights is not None
        weight_update = [
            new_layer_weights - old_layer_weights
            for old_layer_weights, new_layer_weights in zip(self.current_weights, parameters)
        ]
        # return clipped parameters and clipping bit
        return self.clip_parameters(weight_update)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        This function should perform clipping through compute_weight_update_and_clip and store the clipping bit
        as the last entry in the NDArrays
        """
        raise NotImplementedError
