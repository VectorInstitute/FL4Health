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
    strat = FedProx(initial_parameters=params, adaptive_proximal_weight=True, proximal_weight=initial_proximal_weight)
    strat.previous_loss = initial_loss

    new_loss: float = strat.previous_loss + 1

    strat._maybe_update_proximal_weight_param(new_loss)

    assert strat.proximal_weight == initial_proximal_weight + strat.proximal_weight_delta


def test_maybe_update_proximal_weight_param_decrease() -> None:
    layer_size = 10
    num_layers = 5
    initial_proximal_weight = 0.1
    initial_loss = 10
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strat = FedProx(initial_parameters=params, adaptive_proximal_weight=True, proximal_weight=initial_proximal_weight)
    strat.previous_loss = initial_loss

    for c in range(strat.proximal_weight_patience):
        pre_proximal_weight_patience_counter = strat.proximal_weight_patience_counter
        new_loss: float = strat.previous_loss - 1
        strat._maybe_update_proximal_weight_param(new_loss)
        if c == strat.proximal_weight_patience - 1:
            assert strat.proximal_weight == initial_proximal_weight - strat.proximal_weight_delta
            assert strat.proximal_weight_patience_counter == 0
        else:
            assert strat.proximal_weight == initial_proximal_weight
            assert strat.proximal_weight_patience_counter == pre_proximal_weight_patience_counter + 1
