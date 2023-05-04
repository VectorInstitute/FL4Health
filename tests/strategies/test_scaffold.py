import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.scaffold import Scaffold


def test_aggregate() -> None:
    layer_size = 10
    num_layers = 5
    num_clients = 3
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strat = Scaffold(initial_parameters=params)

    client_ndarrays = [[ndarray * (client_num + 1) for ndarray in ndarrays] for client_num in range(num_clients)]
    new_ndarrays = strat.aggregate(client_ndarrays)

    correct_ndarrays = [[ndarray * 2 for ndarray in ndarrays] for _ in range(num_clients)]

    for new_ndarray, correct_ndarray in zip(new_ndarrays, correct_ndarrays):
        assert (new_ndarray == correct_ndarray).all()


def test_compute_updated_parameters() -> None:

    layer_size = 10
    num_layers = 5
    learning_rate = 0.1
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strat = Scaffold(initial_parameters=params)

    original_params: NDArrays = [np.ones((layer_size)) * 3 for _ in range(num_layers)]
    param_updates: NDArrays = [np.ones((layer_size)) * 10 for _ in range(num_layers)]

    updated_params = strat.compute_updated_parameters(learning_rate, original_params, param_updates)
    correct_updated_params = [np.ones_like(original_param) * 4.0 for original_param in original_params]

    for updated_param, correct_updated_param in zip(updated_params, correct_updated_params):
        assert (updated_param == correct_updated_param).all()
