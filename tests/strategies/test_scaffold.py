import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.scaffold import Scaffold


def test_aggregate() -> None:
    layer_size = 10
    num_layers = 5
    num_clients = 3
    num_samples = 50
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strat = Scaffold(initial_parameters=params)

    client_ndarrays = [[ndarray * (client_num + 1) for ndarray in ndarrays] for client_num in range(num_clients)]
    results = [(ndarrays, num_samples) for ndarrays in client_ndarrays]
    new_ndarrays = strat.aggregate(results)

    correct_ndarrays = [[ndarray * 2 for ndarray in ndarrays] for _ in range(num_clients)]

    for new_ndarray, correct_ndarray in zip(new_ndarrays, correct_ndarrays):
        assert (new_ndarray == correct_ndarray).all()
