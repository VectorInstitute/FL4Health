import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.fedprox import FedProx


def test_maybe_update_proximal_weight_param_increase() -> None:
    layer_size = 10
    num_layers = 5
    initial_proximal_weight = 0.1
    initial_loss = 10
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedProx(
        initial_parameters=params, adaptive_proximal_weight=True, proximal_weight=initial_proximal_weight
    )
    strategy.previous_loss = initial_loss

    new_loss: float = strategy.previous_loss + 1

    strategy._maybe_update_proximal_weight_param(new_loss)

    assert strategy.proximal_weight == initial_proximal_weight + strategy.proximal_weight_delta
    assert strategy.proximal_weight_patience_counter == 0


def test_maybe_update_proximal_weight_param_decrease() -> None:
    layer_size = 10
    num_layers = 5
    initial_proximal_weight = 0.1
    initial_loss = 10
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedProx(
        initial_parameters=params, adaptive_proximal_weight=True, proximal_weight=initial_proximal_weight
    )
    strategy.previous_loss = initial_loss

    for c in range(strategy.proximal_weight_patience):
        pre_proximal_weight_patience_counter = strategy.proximal_weight_patience_counter
        new_loss: float = strategy.previous_loss - 1
        strategy._maybe_update_proximal_weight_param(new_loss)
        if c == strategy.proximal_weight_patience - 1:
            assert strategy.proximal_weight == initial_proximal_weight - strategy.proximal_weight_delta
            assert strategy.proximal_weight_patience_counter == 0
        else:
            assert strategy.proximal_weight == initial_proximal_weight
            assert strategy.proximal_weight_patience_counter == pre_proximal_weight_patience_counter + 1
