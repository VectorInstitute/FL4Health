import numpy as np
import pytest
from flwr.common import Parameters

from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM

strategy = ClientLevelDPFedAvgM(
    initial_parameters=Parameters([], ""),
    adaptive_clipping=True,
    server_learning_rate=0.5,
    clipping_learning_rate=0.5,
    weight_noise_multiplier=2.0,
    clipping_noise_mutliplier=5.0,
)


def test_modify_noise_multiplier() -> None:
    assert pytest.approx(strategy.modify_noise_multiplier(), abs=0.0001) == 2.0412


def test_update_with_momentum() -> None:
    np.random.seed(42)
    n_layers = 4
    layer_updates = [np.random.rand(2, 3) for _ in range(n_layers)]
    strategy.calculate_update_with_momentum(layer_updates)
    assert strategy.m_t is not None
    assert np.array_equal(strategy.m_t, layer_updates)

    # should be the same since the update is a weighted average using same update
    strategy.calculate_update_with_momentum(layer_updates)
    target = [(1.0 + strategy.beta) * layer for layer in layer_updates]
    assert strategy.m_t is not None
    assert np.allclose(strategy.m_t, target)

    new_layer_updates = [np.random.rand(2, 3) for _ in range(n_layers)]
    strategy.calculate_update_with_momentum(new_layer_updates)
    assert strategy.m_t is not None
    assert pytest.approx(strategy.m_t[0][0, 0], abs=0.00001) == 1.096533
    assert pytest.approx(strategy.m_t[2][1, 1], abs=0.00001) == 0.642292


def test_calculate_clipping_update() -> None:
    np.random.seed(42)
    clipping_bits = [np.array([4.23])]
    strategy.update_clipping_bound(clipping_bits)
    assert pytest.approx(strategy.clipping_bound, abs=0.00001) == 0.00447446

    # verify updating a second time produces state update
    strategy.update_clipping_bound(clipping_bits)
    assert pytest.approx(strategy.clipping_bound, abs=0.00001) == 0.000979264


def test_split_model_weights_and_clipping_bits() -> None:
    np.random.seed(42)
    n_layers = 4
    n_clients = 3
    n_client_datapoints = 10
    weight_results = [
        (
            [np.random.rand(2, 3) for _ in range(n_layers)] + [np.random.binomial(1, 0.5, 1).astype(float)],
            n_client_datapoints,
        )
        for _ in range(n_clients)
    ]
    weights_only, clip_bits_only = strategy.split_model_weights_and_clipping_bits(weight_results)
    assert np.array_equal(clip_bits_only, [np.array([0]), np.array([0]), np.array([1])])
    for i in range(n_layers):
        layer_list = weight_results[0][0]
        assert np.array_equal(weights_only[0][0][i], layer_list[i])

    for i in range(n_layers):
        layer_list = weight_results[2][0]
        assert np.array_equal(weights_only[2][0][i], layer_list[i])
