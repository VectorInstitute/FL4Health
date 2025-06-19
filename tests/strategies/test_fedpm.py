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

    # Check that the beta parameters have been correctly updated
    # during the first round.
    layer1_alpha = np.full(shape=(3, 3), fill_value=2)
    np.fill_diagonal(layer1_alpha, 3)
    layer1_beta = np.full(shape=(3, 3), fill_value=2)
    np.fill_diagonal(layer1_beta, 1)
    layer3_beta = np.full((5, 5), 3)
    np.fill_diagonal(layer3_beta, 2)
    layer3_alpha = np.ones((5, 5))
    np.fill_diagonal(layer3_alpha, 2)
    expected_beta_parameter_vals = {
        "layer1": (layer1_alpha, layer1_beta),
        "layer2": (np.full((4, 4), 3), np.full((4, 4), 2)),
        "layer3": (layer3_alpha, layer3_beta),
        "layer4": (np.full((6, 6), 2), np.ones((6, 6))),
    }
    assert expected_beta_parameter_vals.keys() == strategy.beta_parameters.keys()
    for layer_name, expected_val in expected_beta_parameter_vals.items():
        alpha, beta = strategy.beta_parameters[layer_name]
        alpha_expected, beta_expected = expected_val
        assert (alpha == alpha_expected).all()
        assert (beta == beta_expected).all()

    # Test that bayesian aggregation works properly in the first round
    # when the beta parameters are initialized on the spot.
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
    for layer_name, layer_val in expected_result.items():
        assert (layer_val == aggregation_result[layer_name]).all()

    # Test that aggregation produces the expected result in the second round
    # using the beta parameters from the first round.
    aggregation_result_second_round = strategy.aggregate_bayesian(results=fit_results)

    layer1_expected_second_round = np.full((3, 3), 0.5)
    np.fill_diagonal(layer1_expected_second_round, 1)
    layer3_expected_second_round = np.zeros((5, 5))
    np.fill_diagonal(layer3_expected_second_round, 0.5)

    expected_result_second_round = {
        "layer1": layer1_expected_second_round,
        "layer2": np.full((4, 4), 2 / 3),
        "layer3": layer3_expected_second_round,
        "layer4": np.ones((6, 6)),
    }
    assert aggregation_result_second_round.keys() == expected_result_second_round.keys()
    for layer_name, layer_val in expected_result_second_round.items():
        assert (layer_val == aggregation_result_second_round[layer_name]).all()

    # Finally, test that resetting beta parameters works properly
    strategy.reset_beta_priors()
    for _, (alpha, beta) in strategy.beta_parameters.items():
        assert (alpha == 1).all() and (beta == 1).all()
