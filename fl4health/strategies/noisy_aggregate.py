from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArray, NDArrays


def add_noise_to_array(layer: NDArray, noise_std_dev: float, denominator: int) -> NDArray:
    # elementwise noise addition and normalization
    layer_noise = np.random.normal(0.0, noise_std_dev, layer.shape)
    return (1.0 / denominator) * (layer + layer_noise)


def add_noise_to_ndarrays(client_model_updates: List[NDArrays], sigma: float, n_clients: int) -> NDArrays:
    layer_sums: NDArrays = [
        add_noise_to_array(reduce(np.add, layer_updates), sigma, n_clients)
        for layer_updates in zip(*client_model_updates)
    ]
    return layer_sums


def gaussian_noisy_unweighted_aggregate(
    results: List[Tuple[NDArrays, int]], noise_multiplier: float, clipping_bound: float
) -> NDArrays:
    """Compute unweighted average of weights. Apply gaussian noise to the sum of.

    Parameters
    ----------
    reults : List[Tuple[NDArrays, int]]
        List of tuples containing the model updates and the number of samples for each client.
    noise_multiplier : float
        The multiplier on the clipping bound to determine the std of noise applied to weight updates.
    clipping_bound : float
        The clipping bound applied to client model updates.
    Returns:
    --------
    layer_sums : NDArrays
        Model update for a given round.
    """
    n_clients = len(results)
    # dropping number of data points component
    client_model_updates = [ndarrays for ndarrays, _ in results]
    sigma = noise_multiplier * clipping_bound
    layer_sums = add_noise_to_ndarrays(client_model_updates, sigma, n_clients)
    return layer_sums


def gaussian_noisy_weighted_aggregate(
    results: List[Tuple[NDArrays, int]],
    noise_multiplier: float,
    clipping_bound: float,
    fraction_fit: float,
    per_client_example_cap: float,
    total_client_weight: float,
) -> NDArrays:
    """Compute weighted average of weights. Apply gaussian noise to the sum of.

    Weighted Implementation based on https://arxiv.org/pdf/1710.06963.pdf

    Parameters
    ----------
    reults : List[Tuple[NDArrays, int]]
        List of tuples containing the model updates and the number of samples for each client.
    noise_multiplier : float
        The multiplier on the clipping bound to determine the std of noise applied to weight updates.
    clipping_bound : float
        The clipping bound applied to client model updates.
    fraction_fit : float, optional
        Fraction of clients sampled each round.
    per_client_example_cap : float, optional
        The maximum number samples per client.
    total_client_weight : float, optional
        The total client weight across samples.

    Returns:
    --------
    layer_sums : NDArrays
        Model update for a given round.
    """

    n_clients = len(results)
    client_model_updates: List[NDArrays] = []
    client_n_points: List[int] = []
    for weights, n_points in results:
        client_model_updates.append(weights)
        client_n_points.append(n_points)

    # Calculate coefs (w_k) by taking the minimum of the sample counts divdied by example cap and 1
    client_coefs = [min((n_points / per_client_example_cap, 1.0)) for n_points in client_n_points]

    # Scale coefs by total expected client weight
    client_coefs_scaled = [coef / (fraction_fit * total_client_weight) for coef in client_coefs]

    # Scale updates by coef for each client
    client_model_updates = [
        [layer_update * client_coef for layer_update in client_model_update]
        for client_model_update, client_coef in zip(client_model_updates, client_coefs_scaled)
    ]  # Calculate model updates as linear combination of updates

    # Update clipping bound as max(w_k) * clipping bound
    # We only require w_k * update is bounded
    # Refer to the footnote on page 4 in https://arxiv.org/pdf/1710.06963.pdf
    updated_clipping_bound = clipping_bound * max(client_coefs)

    sigma = (noise_multiplier * updated_clipping_bound) / fraction_fit
    layer_sums = add_noise_to_ndarrays(client_model_updates, sigma, n_clients)

    return layer_sums


def gaussian_noisy_aggregate_clipping_bits(bits: NDArrays, noise_std_dev: float) -> float:
    n_clients = len(bits)
    bit_sum = reduce(np.add, bits)
    assert bit_sum.shape == (1,)
    noised_bit_sum = add_noise_to_array(bit_sum, noise_std_dev, n_clients)
    return float(noised_bit_sum)
