from functools import reduce

import numpy as np
import pytest

from fl4health.strategies.noisy_aggregate import (
    add_noise_to_array,
    gaussian_noisy_aggregate_clipping_bits,
    gaussian_noisy_unweighted_aggregate,
    gaussian_noisy_weighted_aggregate,
)


def test_add_noise_to_layer() -> None:
    np.random.seed(42)
    layer = np.random.rand(2, 3, 4, 5)
    noised_layer = add_noise_to_array(layer, 1.0, 25)
    assert noised_layer.shape == (2, 3, 4, 5)

    layer = np.random.rand(2, 2)
    noised_layer = add_noise_to_array(layer, 1.0, 25)
    assert pytest.approx(noised_layer[0, 0], abs=0.000001) == -0.002834
    assert pytest.approx(noised_layer[1, 0], abs=0.000001) == 0.0161857


def test_gaussian_noisy_aggregation() -> None:
    np.random.seed(42)
    layer_shape = (2, 3, 4, 5)
    n_clients = 2
    n_layers = 4
    datapoints_per_client = 10
    layers = [
        ([(np.random.rand(*layer_shape)) for _ in range(n_layers)], datapoints_per_client) for _ in range(n_clients)
    ]
    noised_layers = gaussian_noisy_unweighted_aggregate(
        results=layers,
        noise_multiplier=1.0,
        clipping_bound=2.0,
    )
    for i in range(n_layers):
        assert noised_layers[i].shape == layer_shape


def test_gaussian_noisy_aggregate_clipping_bits() -> None:
    np.random.seed(42)
    client_bits = [np.array(1), np.array(1), np.array(0), np.array(1)]
    noised_bit_sum = gaussian_noisy_aggregate_clipping_bits(client_bits, 1.0)
    assert pytest.approx(noised_bit_sum, abs=0.001) == 0.874

    # Ensure invalid shape of clipping bits throws assertion error
    invalid_client_bits = [np.array([1]), np.array([1]), np.array([0]), np.array([1])]
    with pytest.raises(AssertionError):
        _ = gaussian_noisy_aggregate_clipping_bits(invalid_client_bits, 1.0)


def test_weighted_gaussian_noisy_aggregation_shape() -> None:
    np.random.seed(42)
    layer_shape = (2, 3, 4, 5)
    n_clients = 2
    n_layers = 4
    datapoints_per_client = 10
    total_datapoints = datapoints_per_client * n_clients
    layers = [
        ([(np.random.rand(*layer_shape)) for _ in range(n_layers)], datapoints_per_client) for _ in range(n_clients)
    ]
    noised_layers = gaussian_noisy_weighted_aggregate(
        results=layers,
        noise_multiplier=1.0,
        clipping_bound=2.0,
        fraction_fit=1.0,
        per_client_example_cap=float(total_datapoints),
        total_client_weight=1.0,
    )

    for i in range(n_layers):
        assert noised_layers[i].shape == layer_shape


def test_weighted_gaussian_noisy_aggregation_value() -> None:
    layer_shape = (4, 4)
    n_clients = 2
    n_layers = 2
    datapoints_per_client = [25, 75]
    total_datapoints = sum(datapoints_per_client)
    noise_multiplier = 1.0
    clipping_bound = 2.0
    fraction_fit = 1.0

    layers = [
        ([(np.random.rand(*layer_shape)) for _ in range(n_layers)], n_points)
        for _, n_points in zip(range(n_clients), datapoints_per_client)
    ]

    client_1_weights = list(layers[0][0])
    client_2_weights = list(layers[1][0])

    client_1_coef = datapoints_per_client[0] / total_datapoints
    client_2_coef = datapoints_per_client[1] / total_datapoints
    updated_clipping_bound = max(client_1_coef, client_2_coef) * clipping_bound
    sigma = (noise_multiplier * updated_clipping_bound) / fraction_fit

    np.random.seed(42)

    noised_layers_gt = []
    for client_1_layer_weights, client_2_layer_weights in zip(client_1_weights, client_2_weights):
        client_1_layer_weights_ = client_1_coef * client_1_layer_weights
        client_2_layer_weights_ = client_2_coef * client_2_layer_weights
        layer_weights = [client_1_layer_weights_, client_2_layer_weights_]
        updated_layer_weights = add_noise_to_array(reduce(np.add, layer_weights), sigma, n_clients)
        noised_layers_gt.append(updated_layer_weights)

    np.random.seed(42)
    noised_layers = gaussian_noisy_weighted_aggregate(
        results=layers,
        noise_multiplier=1.0,
        clipping_bound=2.0,
        fraction_fit=1.0,
        per_client_example_cap=float(total_datapoints),
        total_client_weight=1.0,
    )

    for noised_layer_gt, noised_layer in zip(noised_layers_gt, noised_layers):
        assert np.allclose(noised_layer_gt, noised_layer)
