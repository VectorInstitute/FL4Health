import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters

from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer


client0_res = [np.ones((3, 3)), np.ones((4, 4))] + [np.array(["layer1", "layer2"])]
client1_res = [np.full((4, 4), 2)] + [np.array(["layer2"])]
client2_res = [np.full((3, 3), 3), np.full((5, 5), 3)] + [np.array(["layer1", "layer3"])]
client3_res = [np.full((4, 4), 4), np.full((6, 6), 4)] + [np.array(["layer2", "layer4"])]
clients_res = [client0_res, client1_res, client2_res, client3_res]
client_train_sizes = [50, 50, 100, 200]
total_train_size = sum(client_train_sizes)


def test_aggregate() -> None:
    layer_size = 10
    num_layers = 5
    ndarrays: NDArrays = [np.ones((layer_size)) for _ in range(num_layers)]
    params: Parameters = ndarrays_to_parameters(ndarrays)
    strategy = FedAvgDynamicLayer(initial_parameters=params)

    aggregate_input = list(zip(clients_res, client_train_sizes))
    aggregate_result = strategy.aggregate(aggregate_input)

    expected_result = {
        "layer1": (client_train_sizes[0] * np.ones((3, 3)) + client_train_sizes[2] * np.full((3, 3), 3))
        / (client_train_sizes[0] + client_train_sizes[2]),
        "layer2": (
            client_train_sizes[0] * np.ones((4, 4))
            + client_train_sizes[1] * np.full((4, 4), 2)
            + client_train_sizes[3] * np.full((4, 4), 4)
        )
        / (client_train_sizes[0] + client_train_sizes[1] + client_train_sizes[3]),
        "layer3": np.full((5, 5), 3),
        "layer4": np.full((6, 6), 4),
    }

    assert expected_result.keys() == aggregate_result.keys()
    for key, val in expected_result.items():
        assert (val == aggregate_result[key]).all()
