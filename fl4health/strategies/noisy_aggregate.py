from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArray, NDArrays


def add_noise_to_array(layer: NDArray, noise_std_dev: float, denominator: int) -> NDArray:
    # elementwise noise addition and normalization
    layer_noise = np.random.normal(0.0, noise_std_dev, layer.shape)
    return (1.0 / denominator) * (layer + layer_noise)


def gaussian_noisy_aggregate(
    results: List[Tuple[NDArrays, int]], noise_multiplier: float, clipping_bound: float, is_weighted: bool = False
) -> NDArrays:
    """Compute weighted or unweighted average of weights. Apply gaussian noise to the sum of"""
    if is_weighted:
        # TODO: Add this capability
        raise NotImplementedError
    else:
        n_clients = len(results)
        # dropping number of data points component
        client_model_weights = [ndarrays for ndarrays, _ in results]
        layer_sums: NDArrays = [
            add_noise_to_array(reduce(np.add, layer_updates), noise_multiplier * clipping_bound, n_clients)
            for layer_updates in zip(*client_model_weights)
        ]
        return layer_sums


def gaussian_noisy_aggregate_clipping_bits(bits: NDArrays, noise_std_dev: float) -> float:
    n_clients = len(bits)
    bit_sum = reduce(np.add, bits)
    assert bit_sum.shape == (1,)
    noised_bit_sum = add_noise_to_array(bit_sum, noise_std_dev, n_clients)
    return float(noised_bit_sum)
