import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


def test_maybe_update_loss_weight_param_increase() -> None:
    layer_size = 10
    num_layers = 5
    initial_loss_weight = 0.1
    initial_loss = 10
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedAvgWithAdaptiveConstraint(
        initial_parameters=params, adapt_loss_weight=True, initial_loss_weight=initial_loss_weight
    )
    strategy.previous_loss = initial_loss

    new_loss: float = strategy.previous_loss + 1

    strategy._maybe_update_constraint_weight_param(new_loss)

    assert strategy.loss_weight == initial_loss_weight + strategy.loss_weight_delta
    assert strategy.loss_weight_patience_counter == 0


def test_maybe_update_loss_weight_param_decrease() -> None:
    layer_size = 10
    num_layers = 5
    initial_loss_weight = 0.1
    initial_loss = 10
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedAvgWithAdaptiveConstraint(
        initial_parameters=params, adapt_loss_weight=True, initial_loss_weight=initial_loss_weight
    )
    strategy.previous_loss = initial_loss

    for c in range(strategy.loss_weight_patience):
        pre_loss_weight_patience_counter = strategy.loss_weight_patience_counter
        new_loss: float = strategy.previous_loss - 1
        strategy._maybe_update_constraint_weight_param(new_loss)
        if c == strategy.loss_weight_patience - 1:
            assert strategy.loss_weight == initial_loss_weight - strategy.loss_weight_delta
            assert strategy.loss_weight_patience_counter == 0
        else:
            assert strategy.loss_weight == initial_loss_weight
            assert strategy.loss_weight_patience_counter == pre_loss_weight_patience_counter + 1
