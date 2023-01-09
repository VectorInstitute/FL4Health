import numpy as np
import pytest

from fl4health.strategies.noisy_aggregate import (
    add_noise_to_array,
    gaussian_noisy_aggregate,
    gaussian_noisy_aggregate_clipping_bits,
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
    noised_layers = gaussian_noisy_aggregate(layers, 1.0, 2.0, datapoints_per_client * n_clients, 1.0)
    for i in range(n_layers):
        assert noised_layers[i].shape == layer_shape


def test_gaussian_noisy_aggregate_clipping_bits() -> None:
    np.random.seed(42)
    client_bits = [np.array([1]), np.array([1]), np.array([0]), np.array([1])]
    noised_bit_sum = gaussian_noisy_aggregate_clipping_bits(client_bits, 1.0)
    assert pytest.approx(noised_bit_sum, abs=0.001) == 0.874


def test_weighted_gaussian_noisy_aggregation() -> None:
    np.random.seed(42)
    layer_shape = (2, 3, 4, 5)
    n_clients = 2
    n_layers = 4
    datapoints_per_client = 10
    layers = [
        ([(np.random.rand(*layer_shape)) for _ in range(n_layers)], datapoints_per_client) for _ in range(n_clients)
    ]
    noised_layers = gaussian_noisy_aggregate(
        layers, 1.0, 2.0, datapoints_per_client * n_clients, 1.0, is_weighted=True
    )
    for i in range(n_layers):
        assert noised_layers[i].shape == layer_shape
