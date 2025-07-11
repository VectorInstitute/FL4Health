import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fl4health.strategies.client_dp_fedavgm import ClientLevelDPFedAvgM
from fl4health.strategies.fedavg_with_adaptive_constraint import FedAvgWithAdaptiveConstraint


client_dp_strategy = ClientLevelDPFedAvgM(
    initial_parameters=None,
    adaptive_clipping=True,
    server_learning_rate=0.5,
    clipping_learning_rate=0.5,
    weight_noise_multiplier=2.0,
    clipping_noise_multiplier=5.0,
)

adapt_constraint_strategy = FedAvgWithAdaptiveConstraint(
    initial_parameters=None, adapt_loss_weight=True, initial_loss_weight=1.0
)

nd_arrays = [np.array([[0.1, 0.2], [1, 2]])]


def test_client_level_dp_fedavgm() -> None:
    parameters = ndarrays_to_parameters(nd_arrays)
    client_dp_strategy.add_auxiliary_information(parameters)

    # Current weights set without the clipping bound
    assert np.allclose(client_dp_strategy.current_weights, nd_arrays)

    # Parameters have been modified to include clipping bound
    target_nd_arrays = parameters_to_ndarrays(parameters)
    assert np.allclose(target_nd_arrays[0], nd_arrays)
    assert np.allclose(target_nd_arrays[1], np.array([0.1]))


def test_fedavg_with_adaptive_constraint() -> None:
    parameters = ndarrays_to_parameters(nd_arrays)
    adapt_constraint_strategy.add_auxiliary_information(parameters)

    # Parameters have been modified to include loss weight
    target_nd_arrays = parameters_to_ndarrays(parameters)
    assert np.allclose(target_nd_arrays[0], nd_arrays)
    assert np.allclose(target_nd_arrays[1], np.array([1.0]))
