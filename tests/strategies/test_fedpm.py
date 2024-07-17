import numpy as np

from fl4health.strategies.fedpm import FedPm

client0_res = [np.identity(3), np.ones((4, 4))] + [np.array(["layer1", "layer2"])]
client1_res = [np.ones((4, 4)), np.zeros((5, 5))] + [np.array(["layer2", "layer3"])]
client2_res = [np.ones((3, 3)), np.identity(5)] + [np.array(["layer1", "layer3"])]
client3_res = [np.zeros((4, 4)), np.ones((6, 6))] + [np.array(["layer2", "layer4"])]
clients_res = [client0_res, client1_res, client2_res, client3_res]
client_train_sizes = [50, 50, 100, 200]
fit_results = [(client_res, n_train) for client_res, n_train in zip(clients_res, client_train_sizes)]


def test_bayesian_aggregation() -> None:
    strategy = FedPm()
    aggregation_result = strategy.aggregate_bayesian(results=fit_results)
    layer1_expected = np.full(shape=(3, 3), fill_value=0.5)
    np.fill_diagonal(layer1_expected, 1)
    layer2_expected = np.full(shape=(4, 4), fill_value=2 / 3)
    layer3_expected = np.zeros((5, 5))
    np.fill_diagonal(layer3_expected, 0.5)
    layer4_expected = np.ones((6, 6))
    expected_result = {
        "layer1": layer1_expected,
        "layer2": layer2_expected,
        "layer3": layer3_expected,
        "layer4": layer4_expected,
    }
    assert expected_result.keys() == aggregation_result.keys()
    for layer_name in aggregation_result.keys():
        assert (expected_result[layer_name] == aggregation_result[layer_name]).all()
