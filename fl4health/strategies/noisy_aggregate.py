from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArray, NDArrays


def add_noise_to_array(layer: NDArray, noise_std_dev: float, denominator: int) -> NDArray:
    # elementwise noise addition and normalization
    layer_noise = np.random.normal(0.0, noise_std_dev, layer.shape)
    return (1.0 / denominator) * (layer + layer_noise)


def gaussian_noisy_aggregate(
    results: List[Tuple[NDArrays, int]],
    noise_multiplier: float,
    clipping_bound: float,
    total_samples: int,
    fraction_fit: float,
    is_weighted: bool = False,
) -> NDArrays:
    """Compute weighted or unweighted average of weights. Apply gaussian noise to the sum of"""
    if is_weighted:
        n_clients = len(results)
        client_model_updates, client_n_points = zip(*results)

        # Calculate client coefficients w_k as proportion of total samples
        client_coefs = [(n_points / total_samples) for n_points in client_n_points]
        client_coefs_scaled = [coef / fraction_fit for coef in client_coefs]  # Scale w_k by 1/fraction_fit
        client_model_updates = [
            [layer_update * client_coef for layer_update in client_model_update]
            for client_model_update, client_coef in zip(client_model_updates, client_coefs_scaled)
        ]  # Calculate model updates as linear combination of updates
        updated_clipping_bound = clipping_bound * max(
            client_coefs
        )  # Update clipping bound as max(w_k) * clippint bound
        sigma = (noise_multiplier * updated_clipping_bound) / fraction_fit
    else:
        n_clients = len(results)
        # dropping number of data points component
        client_model_updates = [ndarrays for ndarrays, _ in results]
        sigma = noise_multiplier * clipping_bound

    layer_sums: NDArrays = [
        add_noise_to_array(reduce(np.add, layer_updates), sigma, n_clients)
        for layer_updates in zip(*client_model_updates)
    ]

    return layer_sums


def gaussian_noisy_aggregate_clipping_bits(bits: NDArrays, noise_std_dev: float) -> float:
    n_clients = len(bits)
    bit_sum = reduce(np.add, bits)
    assert bit_sum.shape == (1,)
    noised_bit_sum = add_noise_to_array(bit_sum, noise_std_dev, n_clients)
    return float(noised_bit_sum)
