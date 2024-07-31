import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.scaffold import Scaffold


def test_aggregate() -> None:
    layer_size = 10
    num_layers = 5
    num_clients = 3
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    variates: Parameters = ndarrays_to_parameters([np.zeros_like(variate) for variate in ndarrays])
    strategy = Scaffold(initial_parameters=params, initial_control_variates=variates)

    client_ndarrays = [[ndarray * (client_num + 1) for ndarray in ndarrays] for client_num in range(num_clients)]
    new_ndarrays = strategy.aggregate(client_ndarrays)

    correct_ndarrays = [[ndarray * 2 for ndarray in ndarrays] for _ in range(num_clients)]

    for new_ndarray, correct_ndarray in zip(new_ndarrays, correct_ndarrays):
        assert (new_ndarray == correct_ndarray).all()


def test_compute_updated_parameters() -> None:
    layer_size = 10
    num_layers = 5
    learning_rate = 0.1
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    variates: Parameters = ndarrays_to_parameters([np.zeros_like(variate) for variate in ndarrays])
    strategy = Scaffold(initial_parameters=params, initial_control_variates=variates)

    original_params: NDArrays = [np.ones((layer_size)) * 3 for _ in range(num_layers)]
    param_updates: NDArrays = [np.ones((layer_size)) * 10 for _ in range(num_layers)]

    updated_params = strategy.compute_updated_parameters(learning_rate, original_params, param_updates)
    correct_updated_params = [np.ones_like(original_param) * 4.0 for original_param in original_params]

    for updated_param, correct_updated_param in zip(updated_params, correct_updated_params):
        assert (updated_param == correct_updated_param).all()


def test_compute_updated_weights() -> None:
    layer_size = (100, 10)
    num_layers = 10
    learning_rate = 0.25

    init_server_weights: NDArrays = [np.ones(layer_size) * 6.0 for _ in range(num_layers)]
    init_params = ndarrays_to_parameters(init_server_weights)

    init_variates: Parameters = ndarrays_to_parameters([np.zeros_like(weights) for weights in init_server_weights])
    strategy = Scaffold(
        initial_parameters=init_params, initial_control_variates=init_variates, learning_rate=learning_rate
    )
    new_server_weights: NDArrays = [np.ones_like(weights) * 46.0 for weights in init_server_weights]
    updated_weights = strategy.compute_updated_weights(new_server_weights)

    correct_updated_weights = [np.ones_like(weights) * 16.0 for weights in init_server_weights]

    for updated_weight, correct_updated_weight in zip(updated_weights, correct_updated_weights):
        assert (updated_weight == correct_updated_weight).all()


def test_compute_updated_control_variates() -> None:
    layer_size = (20, 10)
    num_layers = 5

    init_server_weights: NDArrays = [np.ones(layer_size) for _ in range(num_layers)]
    init_params = ndarrays_to_parameters(init_server_weights)

    init_variates: Parameters = ndarrays_to_parameters([np.zeros_like(weights) for weights in init_server_weights])
    strategy = Scaffold(initial_parameters=init_params, initial_control_variates=init_variates, fraction_fit=0.5)
    control_variate_updates: NDArrays = [np.ones_like(weights) * 20.0 for weights in init_server_weights]
    server_control_variates = strategy.compute_updated_control_variates(control_variate_updates)
    correct_server_control_variates = [np.ones_like(weights) * 10.0 for weights in init_server_weights]

    for control_variates, correct_control_variates in zip(server_control_variates, correct_server_control_variates):
        assert (control_variates == correct_control_variates).all()
